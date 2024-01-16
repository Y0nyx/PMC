import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is built with CUDA: ", tf.test.is_built_with_cuda())

