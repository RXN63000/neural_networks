# 1.1 Elasticity and Scalability


Elasticity: Elasticity refers to the ability of a cloud system to dynamically allocate or deallocate resources (compute, storage, or networking) based on demand. In deep learning, elasticity allows seamless scaling up or down of resources during training and inference, ensuring cost efficiency.

Scalability: Scalability refers to the system's ability to handle increasing workloads by adding resources (either vertically or horizontally). In deep learning, scalability allows the handling of large datasets and complex models, and it is essential for efficient training and inference.

# 1.2 Comparison of AWS SageMaker, Google Vertex AI, and Microsoft Azure Machine Learning Studio


AWS SageMaker: Highly customizable, integrates well with AWS, and is ideal for users already using AWS.
Google Vertex AI: Provides excellent AutoML features, access to Google's pre-trained models and TPUs, ideal for users within the Google Cloud ecosystem.
Microsoft Azure ML Studio: Known for strong MLOps capabilities, seamlessly integrates with Azure services, making it ideal for enterprises using Microsoft’s cloud.


# 2. Convolution Operations with Different Parameters


## 2.1 Convolution with Different Strides and Padding

The Python script below performs convolution on a 5x5 input matrix using a 3x3 kernel with varying stride and padding parameters:

Stride = 1, Padding = 'VALID': Output dims shrink as no padding is used.
Stride = 1, Padding = 'SAME': Padding keeps the input and output dims the same.
Stride = 2, Padding = 'VALID': Output is downsampled by a factor of 2 without padding.
Stride = 2, Padding = 'SAME': Output is downsampled but padding ensures the dimensions remain similar.





import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential

input_matrix = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20],
                         [21, 22, 23, 24, 25]])

kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

input_matrix = input_matrix.reshape(1, 5, 5, 1)
kernel = kernel.reshape(3, 3, 1, 1)

def perform_convolution(stride, padding):
    model = Sequential()
    model.add(Conv2D(1, kernel_size=(3, 3), strides=(stride, stride), padding=padding, input_shape=(5, 5, 1), use_bias=False))
    model.layers[0].set_weights([kernel])
    output = model.predict(input_matrix)
    print(f"Stride = {stride}, Padding = '{padding}':\n{output.squeeze()}\n")

perform_convolution(stride=1, padding='valid')
perform_convolution(stride=1, padding='same')
perform_convolution(stride=2, padding='valid')
perform_convolution(stride=2, padding='same')


# 3. CNN Feature Extraction with Filters and Pooling


## 3.1 Edge Detection Using Sobel Filter

This script applies the Sobel filter for edge detection in both the x and y directions. It visualizes the results using Matplotlib:







import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Could not load image.")
    exit()

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

filtered_x = cv2.filter2D(image, -1, sobel_x)
filtered_y = cv2.filter2D(image, -1, sobel_y)

filtered_combined = np.sqrt(np.square(filtered_x) + np.square(filtered_y))
filtered_combined = np.uint8(filtered_combined)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(filtered_x, cmap='gray')
plt.title('Sobel X-direction')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(filtered_y, cmap='gray')
plt.title('Sobel Y-direction')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(filtered_combined, cmap='gray')
plt.title('Combined Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()


## 3.2 Max Pooling and Average Pooling

This script demonstrates Max Pooling and Average Pooling on a random 4x4 matrix using TensorFlow/Keras.






import tensorflow as tf
import numpy as np

input_matrix = np.random.randint(0, 10, size=(4, 4)).astype(np.float32)
input_matrix = input_matrix.reshape(1, 4, 4, 1)

max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
max_pooled_matrix = max_pool(input_matrix).numpy().squeeze()

avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
avg_pooled_matrix = avg_pool(input_matrix).numpy().squeeze()

print("Original Matrix:\n", input_matrix.squeeze())
print("\nMax-Pooled Matrix:\n", max_pooled_matrix)
print("\nAverage-Pooled Matrix:\n", avg_pooled_matrix)


# 4. Implementing and Comparing CNN Architectures


## 4.1 Simplified AlexNet Architecture

This script defines a simplified version of AlexNet using TensorFlow/Keras. It includes several convolutional layers, max pooling, and fully connected layers.





import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=256, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(filters=384, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=384, kernel_size=(3, 3), activation='relu'),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Flatten(),
    Dense(units=4096, activation='relu'),
    Dropout(rate=0.5),
    Dense(units=4096, activation='relu'),
    Dropout(rate=0.5),
    Dense(units=10, activation='softmax')
])

model.summary()


## 4.2 Residual Block and ResNet Architecture

This script defines a residual block and uses it to create a simple ResNet-like model.





import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Add, Input

def residual_block(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = Add()([x, input_tensor])
    x = tf.keras.activations.relu(x)
    return x

def build_resnet(input_shape=(224, 224, 3), num_classes=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = build_resnet()
model.summary()


## Author

Ravi Teja Reddy Nomula
700756300