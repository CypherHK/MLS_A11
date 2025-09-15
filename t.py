
#print(tf.__version__)  # 应该输出类似 2.15.0 或 2.15.1
#print(tf.config.list_physical_devices('GPU'))  

import tensorflow as tf
with open("mnist_model_quantized.tflite","rb") as f:
    q = f.read()
it = tf.lite.Interpreter(model_content=q); it.allocate_tensors()
inp = it.get_input_details()[0]; out = it.get_output_details()[0]
print(inp["dtype"], inp["quantization"])  # 预期: int8, (scale, zero_point)
print(out["dtype"], out["quantization"])  # 预期: int8 或 uint8, 以及量化参数
print(it.get_input_details()[0]["dtype"], it.get_input_details()[0]["quantization"])
print(it.get_output_details()[0]["dtype"], it.get_output_details()[0]["quantization"])


# LoC
cloc part1_tensorflow.py part2_tflite_conversion.py model_inference.cc model_data.cc
