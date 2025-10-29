# MNIST 嵌入式部署项目文档中心

## 📚 文档导航

本项目包含从模型训练到嵌入式部署的完整 TinyML 流程文档。

### 核心文档

| 文档 | 内容概要 | 阅读时长 |
|------|---------|----------|
| [Part 1: 模型训练](./Part1_Model_Training.md) | 使用 TensorFlow/Keras 训练 MNIST CNN 模型 | 15 分钟 |
| [Part 2: 模型转换](./Part2_Model_Conversion.md) | 转换为 TFLite 格式并进行 INT8 量化 | 20 分钟 |
| [Part 3: 嵌入式部署](./Part3_Embedded_Deployment.md) | 使用 TFLite Micro 在嵌入式设备运行 | 25 分钟 |

---

## 🎯 快速开始

### 完整流程概览

```
┌─────────────────────────────────────────────────────────────┐
│                    MNIST 数据集                              │
│               60,000 训练 + 10,000 测试                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Part 1: 模型训练 (TensorFlow 2.16 + Keras 3)               │
│  ────────────────────────────────────────────               │
│  • 构建 CNN 模型 (Conv2D + Dense)                           │
│  • 训练 5 epochs                                            │
│  • 准确率: 98.8%                                            │
│  • 输出: mnist_cnn_model.keras (286 KB)                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Part 2: 模型转换 (TensorFlow 2.15)                         │
│  ────────────────────────────────────────────               │
│  • 导出权重 → mnist_cnn.weights.h5                          │
│  • Float32 转换 → mnist_model.tflite (88 KB)                │
│  • INT8 量化 → mnist_model_quantized.tflite (25 KB)         │
│  • 量化准确率: 97.7% (损失 0.1%)                            │
│  • 压缩比: 11.57x                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Part 3: 嵌入式部署 (TFLite Micro)                          │
│  ────────────────────────────────────────────               │
│  • 转换为 C 数组 (model_data.cc/h)                          │
│  • 生成算子解析器 (gen_micro_mutable_op_resolver.h)         │
│  • 编写推理代码 (model_inference.cc)                        │
│  • 构建可执行文件 (mnist_micro)                             │
│  • 运行时内存: 8.5 KB                                       │
│  • 推理时间: ~3 ms (macOS ARM64)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 项目指标总结

### 模型性能

| 指标 | 值 |
|------|-----|
| 训练准确率 | 98.8% |
| Float32 TFLite 准确率 | 97.74% |
| INT8 量化准确率 | 97.73% |
| 量化精度损失 | < 0.01% |

### 模型大小

| 格式 | 大小 | 压缩比 |
|------|------|--------|
| Keras (.keras) | 285.93 KB | 1.0x (基准) |
| TFLite Float32 | 88.16 KB | 3.24x |
| TFLite INT8 | 24.71 KB | **11.57x** |

### 资源使用

| 资源 | 需求 |
|------|------|
| 模型存储 | 25 KB (Flash) |
| 运行时内存 | 8.5 KB (RAM) |
| 推理时间 (macOS) | ~3 ms |
| 推理时间 (MCU 估算) | ~50-100 ms |

---

## 🏗️ 模型架构

```
Input: (1, 28, 28, 1)
    ↓
Conv2D(8 filters, 3×3, ReLU)      ← 80 参数
    ↓ (26, 26, 8)
MaxPooling2D(2×2)                 ← 0 参数
    ↓ (13, 13, 8)
Flatten()                         ← 0 参数
    ↓ (1352,)
Dense(16, ReLU)                   ← 21,648 参数
    ↓ (16,)
Dense(10, Softmax)                ← 170 参数
    ↓
Output: (10,)                     Total: 21,898 参数
```

---

## 💻 环境配置

### Part 1 环境 (训练)

```bash
conda create -n mls-trans python=3.10
conda activate mls-trans
pip install tensorflow==2.16.0 keras matplotlib numpy
```

### Part 2 环境 (转换)

```bash
conda create -n mls-a11 python=3.10
conda activate mls-a11
pip install tensorflow==2.15.0 numpy
```

### Part 3 环境 (部署)

```bash
# macOS 系统要求
- Clang++ (Apple Silicon 默认)
- Make
- Bazel
- Git
```

---

## 📁 项目文件结构

```
MLS_A11/
├── docs/                          # 📚 本文档目录
│   ├── README.md                  # 文档中心 (本文件)
│   ├── Part1_Model_Training.md    # Part 1 详细文档
│   ├── Part2_Model_Conversion.md  # Part 2 详细文档
│   └── Part3_Embedded_Deployment.md # Part 3 详细文档
│
├── part1_tensorflow.py            # Part 1: 训练脚本
├── mnist_cnn_model.keras          # Part 1: 输出模型
├── mnist_cnn.weights.h5           # Part 1: 导出权重
│
├── part2_tflite_from_weights.py   # Part 2: 转换脚本
├── mnist_model.tflite             # Part 2: Float32 模型
├── mnist_model_quantized.tflite   # Part 2: INT8 量化模型
├── part2_summary.json             # Part 2: 转换报告
│
├── transToArray.sh                # Part 3: 模型转 C 数组脚本
├── find.sh                        # Part 3: 算子解析器生成脚本
├── model_data.cc / model_data.h   # Part 3: 模型 C 数组
├── model_inference.cc             # Part 3: 推理主程序
├── Makefile                       # Part 3: 构建配置
├── mnist_micro                    # Part 3: 可执行文件
│
├── op_resolver/                   # 算子解析器
│   └── gen_micro_mutable_op_resolver.h
│
└── tflite-micro/                  # TFLite Micro 源码
    └── ...
```

---

## 🎓 学习路径建议

### 初学者路径

1. **先阅读总览** (本文档) - 理解整体流程
2. **Part 1 实践** - 训练模型并理解 CNN 基础
3. **Part 2 实践** - 学习模型压缩与量化
4. **Part 3 阅读** - 了解嵌入式部署概念

### 进阶路径

1. **深入 Part 1** - 尝试不同的模型架构
2. **深入 Part 2** - 研究量化算法和性能权衡
3. **深入 Part 3** - 移植到真实的微控制器平台

### 专家路径

1. **优化模型架构** - 减少参数量，提升速度
2. **高级量化技术** - QAT (量化感知训练)
3. **平台特定优化** - CMSIS-NN, SIMD 优化
4. **生产部署** - OTA 更新、模型加密

---

## 🔧 常用命令速查

### Part 1: 训练

```bash
# 激活环境
conda activate mls-trans

# 训练模型
python part1_tensorflow.py

# 导出权重
python -c "import keras; m = keras.saving.load_model('mnist_cnn_model.keras'); m.save_weights('mnist_cnn.weights.h5')"
```

### Part 2: 转换

```bash
# 激活环境
conda activate mls-a11

# 转换模型
python part2_tflite_from_weights.py

# 查看模型信息
python -c "import tensorflow as tf; interpreter = tf.lite.Interpreter('mnist_model_quantized.tflite'); print(interpreter.get_input_details())"
```

### Part 3: 部署

```bash
# 生成依赖文件
bash transToArray.sh
bash find.sh

# 构建
make lib    # 构建 TFLM 库
make all    # 编译推理程序

# 运行
./mnist_micro

# 清理
make clean
```

---

## ⚠️ 常见问题快速索引

### 环境问题

- **Keras 3 导入错误** → [Part1 文档 - 常见问题](./Part1_Model_Training.md#常见问题与解决方案)
- **TFLite 转换失败** → [Part2 文档 - 常见问题](./Part2_Model_Conversion.md#常见问题与解决方案)
- **编译链接错误** → [Part3 文档 - 常见问题](./Part3_Embedded_Deployment.md#常见问题与解决方案)

### 性能问题

- **训练速度慢** → Part1 文档 - 优化建议
- **量化精度下降** → Part2 文档 - 问题 2
- **推理结果错误** → Part3 文档 - 问题 3

### 部署问题

- **内存不足** → Part3 文档 - Tensor Arena 配置
- **Segmentation fault** → Part3 文档 - 问题 2
- **算子未注册** → Part3 文档 - 算子解析器

---

## 🚀 进阶扩展

### 支持的扩展方向

1. **数据增强** - 提升模型泛化能力
   - 旋转、缩放、平移
   - Cutout, Mixup

2. **模型优化** - 减小模型尺寸
   - 知识蒸馏
   - 神经架构搜索 (NAS)
   - 剪枝 (Pruning)

3. **高级量化** - 进一步压缩
   - 量化感知训练 (QAT)
   - 动态量化
   - 混合精度

4. **跨平台部署**
   - STM32 微控制器
   - Arduino
   - ESP32
   - Raspberry Pi Pico

---

## 📚 参考资源

### 官方文档

- [TensorFlow 官方文档](https://www.tensorflow.org/guide)
- [TFLite 转换器](https://www.tensorflow.org/lite/convert)
- [TFLite Micro](https://github.com/tensorflow/tflite-micro)
- [Keras 3 迁移指南](https://keras.io/guides/migrating_to_keras_3/)

### 学习资源

- [TinyML 书籍](https://tinyml.org/)
- [CS231n CNN 课程](https://cs231n.github.io/)
- [量化指南](https://www.tensorflow.org/lite/performance/post_training_quantization)

### 工具文档

- [FlatBuffers](https://google.github.io/flatbuffers/)
- [Bazel 构建系统](https://bazel.build/)
- [CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)

---

## 🤝 贡献与反馈

如果您发现文档中有错误或不清楚的地方，欢迎提出反馈！

### 文档版本

- **当前版本**: 1.0
- **最后更新**: 2025-10-29
- **适用项目版本**: MLS_A11

---

## 📝 许可证

本项目文档遵循项目的开源许可证。

---

**祝您学习愉快！** 🎉

如有疑问，请按照文档顺序逐步实践，并参考各部分的常见问题章节。
