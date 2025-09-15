#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

#include "tensorflow/lite/schema/schema_generated.h"
// 旧版：#include "tensorflow/lite/version.h"  // 已弃用，不再需要
// 旧版：#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"  // 已弃用
#include "tensorflow/lite/micro/micro_log.h"            // MicroPrintf
#include "tensorflow/lite/micro/system_setup.h"         // InitializeTarget()
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"    //  AddXxx() 时可能常用
#include "op_resolver/gen_micro_mutable_op_resolver.h"  // 自动生成的算子头
#include "model_data.h"                                 // g_mnist_model / g_mnist_model_len

// 预分配 Tensor Arena
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static void MakeTestImage(uint8_t img[28*28]) {
  std::memset(img, 0, 28*28);
  for (int y = 4; y < 24; ++y) {
    for (int t = -1; t <= 1; ++t) {
      int x = 14 + t;
      if (x >= 0 && x < 28) img[y*28 + x] = 255;
    }
  }
}

// 把 0~255 灰度按输入张量量化参数量化到 int8
static void QuantizeToInt8(const uint8_t* src_u8, int8_t* dst_i8, float scale, int zero_point) {
  for (int i = 0; i < 28*28; ++i) {
    float x01 = static_cast<float>(src_u8[i]) / 255.0f;
    int q = static_cast<int>(std::round(x01 / scale + static_cast<float>(zero_point)));
    if (q < -128) q = -128; if (q > 127) q = 127;
    dst_i8[i] = static_cast<int8_t>(q);
  }
}

static int CountElems(const TfLiteTensor* t) {
  if (!t || !t->dims || t->dims->size <= 0) return 1;
  int n = 1;
  for (int i = 0; i < t->dims->size; ++i) n *= t->dims->data[i];
  return n;
}
static void PrintTensorInfo(const char* tag, const TfLiteTensor* t) {
  MicroPrintf("%s:", tag);
  if (!t || !t->dims) { MicroPrintf("  <null tensor or dims>"); return; }
  MicroPrintf("  type=%d, bytes=%d", (int)t->type, (int)t->bytes);
  MicroPrintf("  dims(size=%d): [%d, %d, %d, %d]",
              t->dims->size,
              t->dims->size>0 ? t->dims->data[0] : -1,
              t->dims->size>1 ? t->dims->data[1] : -1,
              t->dims->size>2 ? t->dims->data[2] : -1,
              t->dims->size>3 ? t->dims->data[3] : -1);
  MicroPrintf("  scale=%f, zero_point=%d", t->params.scale, t->params.zero_point);
}
static void PrintFlatbufferMeta(const tflite::Model* model) {
  const auto* subgraph = model->subgraphs()->Get(0);
  // 输入张量类型（按 flatbuffer 原始声明）
  if (subgraph->inputs() && subgraph->inputs()->size() > 0) {
    const int in_idx = subgraph->inputs()->Get(0);
    const auto* tin   = subgraph->tensors()->Get(in_idx);
    MicroPrintf("Flatbuffer input tensor index=%d, type=%d", in_idx, (int)tin->type());
  }
  // 打印算子列表（builtin op code）
  if (subgraph->operators()) {
    MicroPrintf("Operators in model (builtin codes):");
    for (unsigned i = 0; i < subgraph->operators()->size(); ++i) {
      auto* op = subgraph->operators()->Get(i);
      int code = -1;
      if (model->operator_codes()) {
        const auto* oc = model->operator_codes()->Get(op->opcode_index());
        code = oc->builtin_code();
      }
      MicroPrintf("  #%u : builtin_code=%d", i, code);
    }
  }
}

int main() {
  MicroPrintf("ENUM CHECK: kTfLiteInt8=%d, schema INT8=%d, schema FLOAT32=%d",
            (int)kTfLiteInt8, (int)tflite::TensorType_INT8, (int)tflite::TensorType_FLOAT32);

  tflite::InitializeTarget();

  // 1) 读取模型
  const tflite::Model* model = tflite::GetModel(g_mnist_model);
  if (!model) {
    MicroPrintf("GetModel failed");
    return 1;
  }
  MicroPrintf("Model buffer length: %u bytes", (unsigned)g_mnist_model_len);
  PrintFlatbufferMeta(model);

  // 2) 只注册模型实际用到的算子：
  // builtin codes: 3(CONV_2D), 17(MAX_POOL_2D), 22(RESHAPE), 9(FULLY_CONNECTED)*2, 25(SOFTMAX)
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  //auto resolver = get_resolver(); 
  // After  construct `resolver` (static MicroMutableOpResolver<5> ...):
  // === Preflight: ensure every builtin op in the model is registered ===
const auto* subgraph = model->subgraphs()->Get(0);
const auto* opcodes  = model->operator_codes();

for (uint32_t i = 0; i < subgraph->operators()->size(); ++i) {
  const auto* op = subgraph->operators()->Get(i);
  const auto* oc = opcodes->Get(op->opcode_index());

  // TFLM uses tflite::BuiltinOperator (from schema_generated.h)
  const auto builtin = static_cast<tflite::BuiltinOperator>(oc->builtin_code());

  
  const TFLMRegistration* reg = resolver.FindOp(builtin);

  if (reg == nullptr || reg->invoke == nullptr) {
    MicroPrintf("ERROR: Missing kernel for builtin=%d",
                static_cast<int>(builtin));
    return 99; // fail fast instead of crashing in Invoke()
  }
}


  // 3) 解释器 + Arena
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed");
    return 2;
  }
  MicroPrintf("Tensor arena used: %u bytes (arena size: %u bytes)",
              (unsigned)interpreter.arena_used_bytes(), (unsigned)kTensorArenaSize);

  TfLiteTensor* input  = interpreter.input(0);
  // TfLiteTensor* output = interpreter.output(0); // 推迟到 Invoke() 之后获取
  // ---- 在写入输入张量之前，先取 FlatBuffer 的输入类型 ----
const auto* sg = model->subgraphs()->Get(0);
const int in_index = sg->inputs()->Get(0);
const auto* tin_fb = sg->tensors()->Get(in_index);
const auto fb_in_type = tin_fb->type();
  if (fb_in_type == tflite::TensorType_INT8 && input->type != kTfLiteInt8) {
  MicroPrintf("FATAL: Tensor layout mismatch. FlatBuffer=INT8(9) but runtime type=%d; "
              "this indicates mixed headers/libraries. Refusing to invoke to avoid crash.",
              (int)input->type);
  return 10;  // 直接退出，避免在 Invoke 内部再崩
}
  // 仅打印输入张量信息
  //PrintTensorInfo("Input Tensor", input);
  MicroPrintf("Input (runtime): type=%d, bytes=%d, scale=%f, zero_point=%d",
            (int)input->type, (int)input->bytes, input->params.scale, input->params.zero_point);

  // 4) 根据输入张量类型，准备输入数据
  uint8_t img_u8[28*28];
  MakeTestImage(img_u8);

  // 4) 根据输入张量类型，准备输入数据（安全写入：严格按 bytes 校验）

// ---- 只根据 FlatBuffer 的真实类型写入，严格按 bytes 校验 ----
if (fb_in_type == tflite::TensorType_INT8) {
  if (input->bytes != 28*28) {
    MicroPrintf("ERROR: INT8 input expects 784 bytes, got %d", (int)input->bytes);
    return 5;
  }
  QuantizeToInt8(img_u8, input->data.int8, input->params.scale, input->params.zero_point);
  MicroPrintf("Wrote INT8 input using scale=%f, zero_point=%d",
              input->params.scale, input->params.zero_point);

} else if (fb_in_type == tflite::TensorType_FLOAT32) {
  if ((input->bytes % sizeof(float)) != 0) {
    MicroPrintf("ERROR: FLOAT32 input but bytes=%d not divisible by 4", (int)input->bytes);
    return 5;
  }
  const int n = input->bytes / (int)sizeof(float);
  for (int i = 0; i < n; ++i) {
    input->data.f[i] = static_cast<float>(img_u8[i]) / 255.0f;
  }
  MicroPrintf("Wrote FLOAT32 input (n=%d)", n);

} else {
  MicroPrintf("ERROR: Unexpected FB input type %d", (int)fb_in_type);
  return 5;
}
  


  // 5) 推理
  if (interpreter.Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return 4;
  }

  // 6) 读取输出（在 Invoke 成功后再安全地获取和打印输出张量）
  MicroPrintf("Invoke successful!");
  TfLiteTensor* output = interpreter.output(0);
  PrintTensorInfo("Output Tensor", output);

  int out_bytes = output->bytes;
  int best_idx = -1;
  float best_score = -1e30f;

  if (out_bytes == 10) {
    // 多半是 INT8/UINT8 概率/对数几率，逐字节 argmax 即可
    const int8_t* p = reinterpret_cast<const int8_t*>(output->data.raw);
    for (int i = 0; i < 10; ++i) {
      int v = (int)p[i];
      if (v > best_score) { best_score = (float)v; best_idx = i; }
    }
    MicroPrintf("Output heuristic: 10x INT8/UINT8 bytes");
  } else if (out_bytes == 40) {
    // 多半是 10x float32
    const float* p = reinterpret_cast<const float*>(output->data.raw);
    for (int i = 0; i < 10; ++i) {
      float v = p[i];
      if (v > best_score) { best_score = v; best_idx = i; }
    }
    MicroPrintf("Output heuristic: 10x FLOAT32");
  } else {
    // 其它大小：尽力而为（逐字节 argmax）
    const uint8_t* p = reinterpret_cast<const uint8_t*>(output->data.raw);
    for (int i = 0; i < out_bytes; ++i) {
      int v = (int)p[i];
      if (v > best_score) { best_score = (float)v; best_idx = i; }
    }
    MicroPrintf("Output heuristic: raw %d bytes", out_bytes);
  }

  MicroPrintf("Predicted class = %d", best_idx);
  return 0;
}

