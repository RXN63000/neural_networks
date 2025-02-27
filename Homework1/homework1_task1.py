#Ravi Teja Reddy Nomula
#700756300


import tensorflow as tf

#Task 1
#Tensor Manipulations & Reshaping

# 1 create a random tensor of shape (4, 6)
tensor_one = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.int32)

#print the random tensor generated
print(tensor_one)
# 2 Find its rank and shape using TensorFlow functions.
tensor_one_shape = tensor_one.shape
print("Shape = ",tensor_one_shape)

# 3 Reshape it into (2, 3, 4) and transpose it to (3, 2, 4).

#Reshape it into (2, 3, 4)
reshaped_tensor = tf.reshape(tensor_one, (2, 3, 4))
print("Reshaped Tensor = ",reshaped_tensor)

#Transpose it to (3, 2, 4).
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("Transposed Tensor = ",transposed_tensor)

# 4 Broadcast a smaller tensor (1, 4) to match the larger tensor and add them.
small_tensor = tf.constant([[1, 2, 3, 4]])
broadcasted_tensor = small_tensor + reshaped_tensor
print("Broadcasted Tensor = ",broadcasted_tensor)