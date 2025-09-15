python tensorflow/tensorflow/lite/python/convert_file_to_c_source.py \
  --input_tflite_file mnist_model_quantized.tflite \
  --output_source_file model_data.cc \
  --output_header_file model_data.h \
  --array_variable_name g_mnist_model
