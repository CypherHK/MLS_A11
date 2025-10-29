# Part 2: 模型转换详细说明文档

## 📌 概述

**目标**: 将 Keras 3 训练的模型转换为 TensorFlow Lite 格式（float32 和 int8 量化版本）

**输入**: 
- `mnist_cnn.weights.h5` (从 Part 1 导出的权重)
- MNIST 测试数据集（用于量化校准）

**输出**:
- `mnist_model.tflite` (float32 版本, ~88 KB)
- `mnist_model_quantized.tflite` (int8 量化版本, ~25 KB)
- `part2_summary.json` (转换总结报告)

---

## 🛠️ 环境要求

### 软件依赖
```bash
# Conda 环境名称: mls-a11
TensorFlow: 2.15.x (注意：必须是 2.15，不能是 2.16)
NumPy: 最新稳定版
Python: 3.9 或 3.10
```

### 环境配置
```bash
conda create -n mls-a11 python=3.10
conda activate mls-a11
pip install tensorflow==2.15.0 numpy
```

### ⚠️ 版本兼容性说明
| 环境 | TensorFlow | Keras | 用途 |
|------|------------|-------|------|
| mls-trans | 2.16.x | 3.x | 模型训练 (Part 1) |
| mls-a11 | 2.15.x | 2.15 (tf.keras) | TFLite 转换 (Part 2) |

**为什么需要两个环境？**
- Keras 3 模型无法直接被 TF 2.15 的 TFLite 转换器识别
- TF 2.15 的 TFLite 转换器更稳定，特别是量化功能

---

## 📂 文件说明

### 主文件: `part2_tflite_from_weights.py`

**文件结构**:
```python
part2_tflite_from_weights.py
├── set_seed()                      # 设置随机种子（可复现性）
├── load_and_preprocess_data()      # 加载 MNIST 数据
├── build_mnist_cnn()               # 重建模型架构
├── representative_data_gen()       # 量化校准数据生成器
├── convert_to_tflite_from_model()  # 转换为 TFLite
├── analyze_model_size()            # 分析模型大小
├── test_tflite_accuracy()          # 测试 TFLite 模型准确率
└── main()                          # 主执行流程
```

---

## 🔄 转换流程详解

### 完整工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│  Step 0: 在 TF 2.16/Keras 3 环境导出权重                    │
│  keras.Model → mnist_cnn.weights.h5                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 切换到 TF 2.15 环境                                 │
│  conda activate mls-a11                                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 重建模型架构 (tf.keras)                             │
│  build_mnist_cnn() → 相同的网络结构                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 加载权重                                            │
│  model.load_weights("mnist_cnn.weights.h5")                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: 转换为 Float32 TFLite                               │
│  TFLiteConverter.from_keras_model(model)                    │
│  → mnist_model.tflite (~88 KB)                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: 转换为 INT8 量化 TFLite                             │
│  + 使用代表性数据集校准                                      │
│  + 设置量化选项                                              │
│  → mnist_model_quantized.tflite (~25 KB)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 6: 评估转换后的模型                                    │
│  - 测试准确率 (float32: 97.74%, int8: 97.73%)               │
│  - 分析模型大小                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 关键步骤详解

### Step 0: 导出权重（在 TF 2.16 环境）

```python
import keras
m = keras.saving.load_model("mnist_cnn_model.keras")
m.save_weights("mnist_cnn.weights.h5")
print("OK: mnist_cnn.weights.h5")
```

**输出文件**: `mnist_cnn.weights.h5`
- **格式**: HDF5
- **内容**: 仅包含权重参数（不含架构）
- **大小**: ~86 KB

---

### Step 1-3: 重建模型并加载权重

```python
def build_mnist_cnn():
    """重建与 Part 1 完全相同的模型架构"""
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(8, 3, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="mnist_cnn")
    return model

model = build_mnist_cnn()
model.load_weights("mnist_cnn.weights.h5")
```

**关键点**:
- ✅ 使用 **Functional API** 构建（更灵活）
- ✅ 架构必须与 Part 1 **完全一致**（层数、参数、激活函数）
- ✅ 不需要 `compile()`（转换时不需要优化器）

---

### Step 4: Float32 TFLite 转换

```python
def convert_to_tflite_from_model(model, quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if not quantize:
        return converter.convert()  # 默认 float32
    # ... 量化逻辑见下节
```

**转换过程**:
1. 创建转换器对象
2. 解析 Keras 模型图
3. 优化计算图（合并算子、常量折叠）
4. 生成 FlatBuffers 格式

**输出文件**: `mnist_model.tflite`
- **大小**: 88.16 KB
- **精度**: Float32
- **准确率**: 97.74%

---

### Step 5: INT8 量化转换（核心）

#### 5.1 什么是量化？

**量化公式**:
```
quantized_value = round(real_value / scale) + zero_point
```

**好处**:
- 📦 模型大小减少 ~75%
- ⚡ 推理速度提升 2-4x（整数运算更快）
- 🔋 功耗降低（嵌入式设备友好）

**代价**:
- 精度损失 ~0.01%（通常可接受）

#### 5.2 代表性数据集

```python
def representative_data_gen(x_train, num_samples=300):
    """生成用于量化校准的数据"""
    n = min(num_samples, x_train.shape[0])
    for i in range(n):
        yield [x_train[i:i+1]]  # 必须是列表，包含单个样本
```

**作用**:
- 转换器通过这些样本统计每层激活值的范围
- 计算最优的 scale 和 zero_point 参数

**推荐数量**: 100-500 个样本（更多不一定更好）

#### 5.3 量化配置

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = lambda: representative_data_gen(x_train, 300)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

**配置详解**:

| 参数 | 值 | 含义 |
|------|-----|------|
| `optimizations` | `[tf.lite.Optimize.DEFAULT]` | 启用默认优化（权重量化） |
| `representative_dataset` | 数据生成器 | 提供校准数据 |
| `target_spec.supported_ops` | `TFLITE_BUILTINS_INT8` | 仅使用 INT8 算子 |
| `inference_input_type` | `tf.int8` | 输入张量为 INT8 |
| `inference_output_type` | `tf.int8` | 输出张量为 INT8 |

**输出文件**: `mnist_model_quantized.tflite`
- **大小**: 24.71 KB
- **精度**: INT8
- **准确率**: 97.73%

---

## 📊 模型分析

### 大小对比

| 模型格式 | 文件大小 | 压缩比 |
|---------|---------|--------|
| Keras (.keras) | 285.93 KB | 1.0x (基准) |
| TFLite Float32 | 88.16 KB | 3.24x |
| TFLite INT8 | 24.71 KB | **11.57x** |

**关键观察**:
- Float32 TFLite 比 Keras 小 3.2 倍（移除训练相关信息）
- INT8 量化进一步压缩 3.6 倍（8 位表示替代 32 位浮点）

### 准确率对比

```json
{
  "accuracy": {
    "tflite_float32": 0.9774,
    "tflite_int8": 0.9773
  }
}
```

**结论**:
- 量化损失仅 **0.01%**（10,000 张测试图像中差 1 张）
- 对于嵌入式部署，这个精度损失**完全可接受**

---

## 🔍 关键代码深度解析

### 1. 量化输入预处理

```python
def _quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """将浮点输入量化为 INT8"""
    if scale == 0: 
        return np.zeros_like(x, dtype=np.int8)
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)
```

**步骤**:
1. `x / scale` → 按比例缩放
2. `+ zero_point` → 偏移到 INT8 范围
3. `np.clip(q, -128, 127)` → 裁剪到 [-128, 127]
4. `astype(np.int8)` → 转换为有符号 8 位整数

**示例**:
```python
# 假设 scale=0.003922, zero_point=-128
# 输入像素值 255 (白色)
q = round(1.0 / 0.003922 + (-128)) 
  = round(255 - 128) 
  = 127  # INT8 最大值
```

### 2. TFLite 推理测试

```python
def test_tflite_accuracy(tflite_model_data, x_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_data)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 获取量化参数
    in_scale = input_details[0].get("quantization", (0.0, 0))[0]
    in_zero = input_details[0].get("quantization", (0.0, 0))[1]
    
    correct = 0
    for i in range(len(x_test)):
        # 量化输入
        x = x_test[i:i+1]
        if input_details[0]["dtype"] == np.int8:
            in_data = _quantize_input(x, float(in_scale), int(in_zero))
        else:
            in_data = x.astype(np.float32)
        
        # 推理
        interpreter.set_tensor(input_details[0]["index"], in_data)
        interpreter.invoke()
        
        # 解析输出
        out = interpreter.get_tensor(output_details[0]["index"])
        pred = int(np.argmax(out))
        correct += int(pred == int(y_test[i]))
    
    return correct / len(x_test)
```

**关键步骤**:
1. 创建 TFLite 解释器
2. 分配张量内存
3. 获取输入/输出元数据（类型、形状、量化参数）
4. 逐样本推理并统计准确率

---

## ⚠️ 常见问题与解决方案

### 问题 1: 转换器报错 "No quantization parameters"

**错误信息**:
```
ValueError: Cannot set tensor: Got value of type INT8 but expected type FLOAT32
```

**原因**: 未提供 representative_dataset

**解决方案**:
```python
# 确保设置代表性数据集
converter.representative_dataset = lambda: representative_data_gen(x_train, 300)
```

### 问题 2: 量化后准确率大幅下降 (>5%)

**可能原因**:
- 代表性数据集不够多样化
- 模型本身对量化敏感

**解决方案**:
```python
# 增加代表性数据集样本数
representative_data_gen(x_train, num_samples=1000)

# 或使用动态范围量化（仅量化权重）
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 不设置 target_spec 和 inference_*_type
```

### 问题 3: 转换时出现 "Non-Converted Ops" 警告

**警告信息**:
```
Summary on the non-converted ops:
 * 7 ARITH ops
```

**说明**: 
- 这是正常的，`arith.constant` 是 MLIR 中间表示的常量操作
- 最终会被优化掉，不影响 TFLite 模型

**验证方法**:
```bash
# 使用 visualize.py 查看模型图
python -m tensorflow.lite.python.visualize mnist_model_quantized.tflite \
    --output_html model_graph.html
```

### 问题 4: 模型文件损坏

**症状**: 无法加载 .tflite 文件

**检查方法**:
```bash
# 验证文件完整性
xxd mnist_model_quantized.tflite | head -1
# 应该看到 TFLite 文件头: 54464c33 (TFL3)
```

---

## 🧪 验证测试

### 1. 基本功能测试

```python
# 加载量化模型
interpreter = tf.lite.Interpreter(model_path="mnist_model_quantized.tflite")
interpreter.allocate_tensors()

# 获取输入输出细节
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Input type:", input_details[0]['dtype'])
print("Output shape:", output_details[0]['shape'])
print("Output type:", output_details[0]['dtype'])
```

**预期输出**:
```
Input shape: [  1  28  28   1]
Input type: <class 'numpy.int8'>
Output shape: [ 1 10]
Output type: <class 'numpy.int8'>
```

### 2. 单样本推理测试

```python
# 准备测试图像（数字"1"）
test_image = x_test[0:1]  # shape: (1, 28, 28, 1)

# 量化输入
scale = input_details[0]['quantization'][0]
zero_point = input_details[0]['quantization'][1]
quantized_input = np.round(test_image / scale + zero_point).astype(np.int8)

# 推理
interpreter.set_tensor(input_details[0]['index'], quantized_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# 解析结果
predicted_class = np.argmax(output)
print(f"Predicted: {predicted_class}, Actual: {y_test[0]}")
```

---

## 📝 运行检查清单

**环境准备**:
- [ ] 确认已安装 TensorFlow 2.15
- [ ] 确认 `mnist_cnn.weights.h5` 文件存在
- [ ] 切换到 `mls-a11` Conda 环境

**执行转换**:
- [ ] 运行 `python part2_tflite_from_weights.py`
- [ ] 检查生成 `mnist_model.tflite` (~88 KB)
- [ ] 检查生成 `mnist_model_quantized.tflite` (~25 KB)
- [ ] 检查生成 `part2_summary.json`

**结果验证**:
- [ ] Float32 模型准确率 > 97%
- [ ] INT8 模型准确率 > 97%
- [ ] 压缩比 > 10x

---

## 🎯 下一步操作

完成 Part 2 后，准备进入 Part 3（嵌入式部署）：

1. **转换模型为 C 数组**:
   ```bash
   bash transToArray.sh
   ```

2. **生成算子解析器**:
   ```bash
   bash find.sh
   ```

3. **编写推理代码**: `model_inference.cc`

---

## 📚 参考资源

- [TFLite 转换器官方文档](https://www.tensorflow.org/lite/convert)
- [TFLite 量化指南](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [FlatBuffers 格式说明](https://google.github.io/flatbuffers/)
- [TFLite 模型优化工具包](https://www.tensorflow.org/model_optimization)
