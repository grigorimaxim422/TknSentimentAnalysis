# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
import tensorflow as tf
print(tf.test.is_built_with_cuda())  # Should return True
print(tf.test.is_gpu_available())    # Should return True (if TensorFlow < 2.10)
print(tf.config.list_physical_devices('GPU'))  # Should show your GPU
