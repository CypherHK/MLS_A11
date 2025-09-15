
"""
Part 2 : Convert from tf.keras model rebuilt in TF 2.15 using weights exported from Keras 3.
- Rebuilds the exact MNIST CNN in tf.keras
- Loads weights from 'mnist_cnn.weights.h5'
- Produces float32 and INT8 
- Compares sizes and evaluates accuracy with the TFLite Interpreter
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_and_preprocess_data():
    """Load MNIST, normalize to [0,1], expand channel dimension."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, -1)   # (60000, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)     # (10000, 28, 28, 1)
    return x_train, y_train, x_test, y_test


# -----------------------------
# Rebuild the exact architecture in tf.keras
# -----------------------------
def build_mnist_cnn():
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(8, 3, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="mnist_cnn")
    return model


# -----------------------------
# Conversion helpers
# -----------------------------
def representative_data_gen(x_train: np.ndarray, num_samples: int = 300):
    n = min(num_samples, x_train.shape[0])
    for i in range(n):
        yield [x_train[i:i+1]]


def convert_to_tflite_from_model(model: keras.Model, quantize: bool = False) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if not quantize:
        return converter.convert()

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    x_train, _, _, _ = load_and_preprocess_data()
    converter.representative_dataset = lambda: representative_data_gen(x_train, num_samples=300)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def analyze_model_size(tf_model_path: str | None, tflite_model_data: bytes, label: str):
    tflite_bytes = len(tflite_model_data)
    def fmt(n):
        if n < 1024: return f"{n} B"
        if n < 1024**2: return f"{n/1024:.2f} KB"
        return f"{n/1024**2:.2f} MB"

    print("\n[Size Analysis]")
    print(f"- {label} (.tflite): {fmt(tflite_bytes)}")
    if tf_model_path and os.path.exists(tf_model_path):
        keras_bytes = os.path.getsize(tf_model_path)
        ratio = keras_bytes / tflite_bytes if tflite_bytes else float('inf')
        print(f"- Keras (.keras): {fmt(keras_bytes)}")
        print(f"- Compression ratio (keras/{label}): {ratio:.2f}x")
    return tflite_bytes


def _quantize_input(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale == 0: return np.zeros_like(x, dtype=np.int8)
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def test_tflite_accuracy(tflite_model_data: bytes, x_test: np.ndarray, y_test: np.ndarray) -> float:
    interpreter = tf.lite.Interpreter(model_content=tflite_model_data)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_idx = input_details[0]["index"]
    out_idx = output_details[0]["index"]
    in_dtype = input_details[0]["dtype"]
    in_scale, in_zero = input_details[0].get("quantization", (0.0, 0))

    # Ensure input shape is [1,28,28,1]
    try:
        interpreter.resize_tensor_input(in_idx, [1,28,28,1])
        interpreter.allocate_tensors()
    except Exception:
        pass

    correct = 0
    for i in range(len(x_test)):
        x = x_test[i:i+1]
        if in_dtype == np.float32:
            in_data = x.astype(np.float32)
        else:
            in_data = _quantize_input(x.astype(np.float32), float(in_scale), int(in_zero))
        interpreter.set_tensor(in_idx, in_data)
        interpreter.invoke()
        out = interpreter.get_tensor(out_idx)
        pred = int(np.argmax(out, axis=1)[0])
        correct += int(pred == int(y_test[i]))
    return correct / len(x_test)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    # 1) Rebuild model in tf.keras and load weights exported from Keras 3
    weights_path = "mnist_cnn.weights.h5"  
    if not os.path.exists(weights_path):
        raise SystemExit(f"ERROR: '{weights_path}' not found. Place it next to this script.")

    model = build_mnist_cnn()
    model.load_weights(weights_path)
    print("Loaded weights into rebuilt tf.keras model.")
    model.summary()

    # 2) Convert to TFLite (float32 + INT8)
    print("\nConverting float32 TFLite...")
    tflite_float = convert_to_tflite_from_model(model, quantize=False)
    with open("mnist_model.tflite", "wb") as f:
        f.write(tflite_float)
    print("Saved: mnist_model.tflite")

    print("Converting INT8 (full integer) TFLite...")
    tflite_int8 = convert_to_tflite_from_model(model, quantize=True)
    with open("mnist_model_quantized.tflite", "wb") as f:
        f.write(tflite_int8)
    print("Saved: mnist_model_quantized.tflite")

    # 3) Size analysis 
    keras_path = "mnist_cnn_model.keras"
    analyze_model_size(keras_path if os.path.exists(keras_path) else None, tflite_float, "tflite-float32")
    analyze_model_size(keras_path if os.path.exists(keras_path) else None, tflite_int8, "tflite-int8")

    # 4) TFLite accuracy evaluation
    print("\nLoading MNIST test data...")
    _, _, x_test, y_test = load_and_preprocess_data()

    print("Evaluating float32 TFLite...")
    acc_f = test_tflite_accuracy(tflite_float, x_test, y_test)
    print(f"TFLite float32 accuracy: {acc_f:.4f}")

    print("Evaluating INT8 TFLite...")
    acc_q = test_tflite_accuracy(tflite_int8, x_test, y_test)
    print(f"TFLite INT8 accuracy: {acc_q:.4f}")

    # 5) Summary JSON
    with open("part2_summary.json", "w") as f:
        json.dump({
            "weights_file": weights_path,
            "tflite_float32_file": "mnist_model.tflite",
            "tflite_int8_file": "mnist_model_quantized.tflite",
            "accuracy": {"tflite_float32": float(acc_f), "tflite_int8": float(acc_q)}
        }, f, indent=2)
    print("\nWrote summary: part2_summary.json")
