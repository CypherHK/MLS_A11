# 1) 进到 tflite-micro 仓库根目录（这里有 MODULE.bazel/WORKSPACE）
cd tflite-micro

# 2) 准备模型路径 & 输出目录
MODEL_DIR="/Users/wangyucheng/Projects/MLS_A11"     #  .tflite 放这里
OUT_DIR="$MODEL_DIR/op_resolverNew"
mkdir -p "$OUT_DIR"

# 自检模型真的在这
ls -lh "$MODEL_DIR/mnist_model_quantized.tflite" || { echo "模型不在 $MODEL_DIR"; exit 1; }

# 3) 在 tflite-micro 仓库根目录运行 bazel run
bazel run //tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver:generate_micro_mutable_op_resolver_from_model -- \
  --common_tflite_path="$MODEL_DIR" \
  --input_tflite_files=mnist_model_quantized.tflite \
  --output_dir="$OUT_DIR"
