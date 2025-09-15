# MNIST Handwritten Digit Recognition: From Keras to TFLite Micro

# **Part 1: Model Training (TensorFlow/Keras)**
    *   In a TensorFlow 2.16 and Keras 3 environment, use part1_tensorflow.py to train a small CNN model for MNIST handwritten digit recognition.
    *   After training, the model is saved as mnist_cnn_model.keras.


# **Part 2: Model Conversion (TensorFlow Lite)**
    *   Due to compatibility issues between Keras 3 and TensorFlow 2.15, first export the weights from the .keras model to a .weights.h5 file.
    *   Then, in a TensorFlow 2.15 environment, use part2_tflite_from_weights.py to rebuild the same model architecture and load the previously exported weights.
    *   Convert the model with loaded weights to two TFLite formats:
        *   `mnist_model.tflite` (float32)
        *   `mnist_model_quantized.tflite` (int8 full integer quantization)
    *   The script will also analyze the size of the converted models and evaluate their accuracy.

## 2.1) In the TF 2.16/Keras 3 environment, export the weights from the .keras model to a .weights.h5 file.
python - <<'PY'
import keras
m = keras.saving.load_model("mnist_cnn_model.keras")
m.save_weights("mnist_cnn.weights.h5")
print("OK: mnist_cnn.weights.h5")
PY

## 2.2) Switch to the TF 2.15 environment (mls-a11)
    *   Use the Part 1 code to rebuild the same tf.keras model structure, then:
python - <<'PY'
import tensorflow as tf
from tensorflow import keras
inputs = keras.Input(shape=(28,28,1), name="input_layer")
x = keras.layers.Conv2D(8, 3, activation="relu", name="conv")(inputs)
x = keras.layers.MaxPooling2D(2, name="pool")(x)
x = keras.layers.Flatten(name="flatten")(x)
x = keras.layers.Dense(16, activation="relu", name="fc")(x)
outputs = keras.layers.Dense(10, activation="softmax", name="pred")(x)
model = keras.Model(inputs, outputs)
model.load_weights("mnist_cnn.weights.h5")
model.summary()
PY
## 2.3） Next, proceed to Part 2’s conversion function and call converter.from_keras_model(model) on the model.

## Output：
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_layer (InputLayer)    [(None, 28, 28, 1)]       0         
                                                                 
 conv (Conv2D)               (None, 26, 26, 8)         80        
                                                                 
 pool (MaxPooling2D)         (None, 13, 13, 8)         0         
                                                                 
 flatten (Flatten)           (None, 1352)              0         
                                                                 
 fc (Dense)                  (None, 16)                21648     
                                                                 
 pred (Dense)                (None, 10)                170       
                                                                 
=================================================================
Total params: 21898 (85.54 KB)
Trainable params: 21898 (85.54 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

# part2 output：
## Converting float32 TFLite...
Summary on the non-converted ops:
---------------------------------
 * Accepted dialects: tfl, builtin, func
 * Non-Converted Ops: 7, Total Ops 16, % non-converted = 43.75 %
 * 7 ARITH ops

- arith.constant:    7 occurrences  (f32: 6, i32: 1)



  (f32: 1)
  (f32: 2)
  (f32: 1)
  (f32: 1)
  (f32: 1)
Saved: mnist_model.tflite
## Converting INT8 (full integer) TFLite...
Saved: mnist_model_quantized.tflite

[Size Analysis]
- tflite-float32 (.tflite): 88.16 KB
- Keras (.keras): 285.93 KB
- Compression ratio (keras/tflite-float32): 3.24x

[Size Analysis]
- tflite-int8 (.tflite): 24.71 KB
- Keras (.keras): 285.93 KB
- Compression ratio (keras/tflite-int8): 11.57x

#  **Part 3: On-device Inference (TFLite Micro)**

- Use the transToArray script to convert the quantized model into C arrays (.cc, .h).
- (In my method, please git clone tensorflow first.)
- Use find.sh to automatically generate the required operator resolver.
- Write the main file (model_inference.cc) and a Makefile, then build.
- Sample inference log: % ./mnist_micro
ENUM CHECK: kTfLiteInt8=9, schema INT8=9, schema FLOAT32=0
Model buffer length: 25304 bytes
Flatbuffer input tensor index=0, type=9
Operators in model (builtin codes):
  #0 : builtin_code=3
  #1 : builtin_code=17
  #2 : builtin_code=22
  #3 : builtin_code=9
  #4 : builtin_code=9
  #5 : builtin_code=25
Tensor arena used: 8480 bytes (arena size: 61440 bytes)
Input (runtime): type=9, bytes=784, scale=0.003922, zero_point=-128
Wrote INT8 input using scale=0.003922, zero_point=-128
Invoke successful!
Output Tensor:
  type=9, bytes=10
  dims(size=2): [1, 10, -1, -1]
  scale=0.003906, zero_point=-128
Output heuristic: 10x INT8/UINT8 bytes
Predicted class = 1

**Inference Log Interpretation:**

The log shows the model loaded successfully, all operators were correctly registered, memory allocation was sufficient, and the final prediction for the test image is “1,” as expected.