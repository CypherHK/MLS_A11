# Part 1: 模型训练详细说明文档

## 📌 概述

**目标**: 使用 TensorFlow 2.16 和 Keras 3 训练一个轻量级 CNN 模型，用于 MNIST 手写数字识别。

**输入**: MNIST 数据集（60,000 训练图像 + 10,000 测试图像）  
**输出**: `mnist_cnn_model.keras` 模型文件

---

## 🛠️ 环境要求

### 软件依赖
```bash
# Conda 环境名称: mls-trans
TensorFlow: 2.16.x
Keras: 3.x
NumPy: 最新稳定版
Matplotlib: 用于可视化训练过程
```

### 安装命令
```bash
conda create -n mls-trans python=3.10
conda activate mls-trans
pip install tensorflow==2.16.0 keras matplotlib numpy
```

---

## 📂 文件说明

### 主文件: `part1_tensorflow.py`

**文件结构**:
```
part1_tensorflow.py
├── create_model()           # 构建 CNN 模型架构
├── load_and_preprocess_data() # 加载并预处理 MNIST 数据
├── train_model()            # 训练模型
├── plot_history()           # 可视化训练历史
└── main()                   # 主执行流程
```

---

## 🏗️ 模型架构详解

### 网络结构

```python
Sequential Model:
┌─────────────────────────────────────────┐
│ Input: (28, 28, 1)                      │
├─────────────────────────────────────────┤
│ Conv2D(8 filters, 3x3, ReLU)            │  ← 80 参数
│   Output: (26, 26, 8)                   │
├─────────────────────────────────────────┤
│ MaxPooling2D(2x2)                       │  ← 0 参数
│   Output: (13, 13, 8)                   │
├─────────────────────────────────────────┤
│ Flatten()                               │  ← 0 参数
│   Output: (1352,)                       │
├─────────────────────────────────────────┤
│ Dense(16, ReLU)                         │  ← 21,648 参数
│   Output: (16,)                         │
├─────────────────────────────────────────┤
│ Dense(10, Softmax)                      │  ← 170 参数
│   Output: (10,)                         │
└─────────────────────────────────────────┘
Total Parameters: 21,898 (85.54 KB)
```

### 层级分析

#### 1. Conv2D 层
```python
keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1))
```
- **作用**: 提取图像的局部特征（边缘、纹理）
- **参数数量**: `(3×3×1 + 1) × 8 = 80`
  - 3×3 卷积核 × 1 输入通道 × 8 输出通道 + 8 偏置
- **输出形状**: (26, 26, 8)
  - 26 = 28 - 3 + 1（valid 填充）

#### 2. MaxPooling2D 层
```python
keras.layers.MaxPooling2D((2, 2))
```
- **作用**: 下采样，减少参数量，提取主要特征
- **输出形状**: (13, 13, 8)
  - 13 = 26 / 2

#### 3. Flatten 层
```python
keras.layers.Flatten()
```
- **作用**: 将 3D 特征图展平为 1D 向量
- **输出形状**: (1352,)
  - 1352 = 13 × 13 × 8

#### 4. Dense(16) 层
```python
keras.layers.Dense(16, activation='relu')
```
- **作用**: 全连接层，学习特征组合
- **参数数量**: `1352 × 16 + 16 = 21,648`

#### 5. Dense(10) 输出层
```python
keras.layers.Dense(10, activation='softmax')
```
- **作用**: 分类层，输出 10 个类别的概率分布
- **参数数量**: `16 × 10 + 10 = 170`

---

## 📊 数据预处理流程

### 1. 加载数据
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```
- **训练集**: 60,000 张 28×28 灰度图像
- **测试集**: 10,000 张 28×28 灰度图像

### 2. 归一化
```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```
- **目的**: 将像素值从 [0, 255] 归一化到 [0, 1]
- **好处**: 加速梯度下降收敛

### 3. 添加通道维度
```python
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28) → (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28) → (10000, 28, 28, 1)
```
- **目的**: CNN 需要显式的通道维度（灰度图为 1 通道）

---

## 🚀 训练配置

### 编译配置
```python
model.compile(
    optimizer='adam',              # 自适应学习率优化器
    loss='sparse_categorical_crossentropy',  # 多分类交叉熵损失
    metrics=['accuracy']           # 监控准确率
)
```

### 训练参数
```python
history = model.fit(
    x_train, y_train,
    epochs=5,                      # 训练 5 轮
    validation_data=(x_test, y_test)  # 每轮结束后在测试集上验证
)
```

---

## 📈 预期训练结果

### 典型输出示例
```
Epoch 1/5
1875/1875 [==============================] - 15s 8ms/step 
    - loss: 0.2314 - accuracy: 0.9327 - val_loss: 0.0894 - val_accuracy: 0.9726
Epoch 2/5
1875/1875 [==============================] - 14s 7ms/step 
    - loss: 0.0766 - accuracy: 0.9763 - val_loss: 0.0656 - val_accuracy: 0.9798
Epoch 3/5
1875/1875 [==============================] - 14s 7ms/step 
    - loss: 0.0550 - accuracy: 0.9826 - val_loss: 0.0591 - val_accuracy: 0.9816
Epoch 4/5
1875/1875 [==============================] - 14s 7ms/step 
    - loss: 0.0436 - accuracy: 0.9859 - val_loss: 0.0553 - val_accuracy: 0.9826
Epoch 5/5
1875/1875 [==============================] - 14s 7ms/step 
    - loss: 0.0361 - accuracy: 0.9882 - val_loss: 0.0530 - val_accuracy: 0.9834
```

### 性能指标
- **最终训练准确率**: ~98.8%
- **最终验证准确率**: ~98.3%
- **训练时间**: 约 70 秒（Apple M 系列芯片）

---

## 📊 可视化分析

### 训练曲线
`plot_history()` 函数生成两个图表：

#### 1. 准确率曲线
- **训练准确率**: 持续上升，从 93% → 98.8%
- **验证准确率**: 从 97% → 98.3%
- **观察**: 无明显过拟合（训练和验证曲线接近）

#### 2. 损失曲线
- **训练损失**: 从 0.23 下降到 0.036
- **验证损失**: 从 0.089 下降到 0.053
- **观察**: 验证损失略高于训练损失，但差距较小

---

## 💾 模型保存

### 输出文件
```python
# 默认情况下，Keras 3 会自动保存为 .keras 格式
model.save("mnist_cnn_model.keras")
```

### 文件信息
- **文件名**: `mnist_cnn_model.keras`
- **文件大小**: ~286 KB
- **格式**: Keras 3 原生格式（基于 ZIP 的 HDF5）
- **包含内容**:
  - 模型架构（JSON 格式）
  - 权重参数（HDF5 格式）
  - 优化器状态
  - 训练配置

---

## 🔍 关键代码解析

### 1. 模型创建函数
```python
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```
**要点**:
- 使用 Sequential API（适合线性堆叠的层）
- 激活函数选择：ReLU（隐藏层）+ Softmax（输出层）
- 损失函数：`sparse_categorical_crossentropy`（标签为整数时使用）

### 2. 数据预处理函数
```python
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, y_train, x_test, y_test
```
**要点**:
- 类型转换：`uint8` → `float32`（避免整数除法问题）
- 归一化：除以 255.0（而非 255，确保浮点运算）
- 维度扩展：`-1` 表示在最后添加维度

---

## ⚠️ 常见问题与解决方案

### 问题 1: Keras 3 导入错误
```
ImportError: cannot import name 'Sequential' from 'keras'
```
**解决方案**:
```bash
# 确保使用 TensorFlow 2.16+ 
pip install tensorflow==2.16.0
# Keras 3 已集成在 TensorFlow 中
```

### 问题 2: 内存不足
```
ResourceExhaustedError: OOM when allocating tensor
```
**解决方案**:
```python
# 减小批次大小
history = model.fit(x_train, y_train, batch_size=32, epochs=5)
```

### 问题 3: 训练速度慢
**优化建议**:
- 使用 GPU（如果可用）
- 减少训练轮次（5 轮通常已足够）
- 使用数据增强时考虑并行加载

---

## 🎯 下一步操作

完成 Part 1 后，需要导出权重以便在 TensorFlow 2.15 环境中使用：

```python
import keras
m = keras.saving.load_model("mnist_cnn_model.keras")
m.save_weights("mnist_cnn.weights.h5")
print("权重已导出: mnist_cnn.weights.h5")
```

**原因**: Keras 3 与 TensorFlow 2.15 的 TFLite 转换器存在兼容性问题，需要先导出权重，再在 TF 2.15 环境中重建模型。

---

## 📝 运行检查清单

- [ ] 确认 Conda 环境为 `mls-trans`
- [ ] 确认 TensorFlow 版本为 2.16.x
- [ ] 确认 Keras 版本为 3.x
- [ ] 运行 `python part1_tensorflow.py`
- [ ] 检查生成的 `mnist_cnn_model.keras` 文件
- [ ] 验证模型大小约为 286 KB
- [ ] 确认训练准确率 > 98%
- [ ] 导出权重到 `mnist_cnn.weights.h5`

---

## 📚 参考资源

- [TensorFlow 官方文档](https://www.tensorflow.org/guide)
- [Keras 3 迁移指南](https://keras.io/guides/migrating_to_keras_3/)
- [MNIST 数据集介绍](http://yann.lecun.com/exdb/mnist/)
- [CNN 架构详解](https://cs231n.github.io/convolutional-networks/)
