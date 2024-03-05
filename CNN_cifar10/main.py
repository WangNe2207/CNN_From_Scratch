import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10 as cf10
from tensorflow.keras.utils import to_categorical
from scipy.signal import correlate2d
from scipy.signal import correlate
from sklearn.metrics import accuracy_score

(train_images, train_labels), (test_images, test_labels) = cf10.load_data()

X_train = train_images[:5000] / 255.0
y_train = train_labels[:5000]

X_test = train_images[5000:10000] / 255.0
y_test = train_labels[5000:10000]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # Create a 3D matrix (cube) with shape (2, 3, 4)
# B = np.zeros((3, 3, 4))
# # Get the first layer (2D matrix) from the 3D matrix
# first_layer = B[0]
#
# # Print the first layer
# print(B.shape)
# print("First layer (2D matrix):")
#
# print(first_layer)


class Convolution:
    def __init__(self, input_shape, filter_size, filter_depth,num_filters):
        input_height, input_width, input_depth = input_shape
        self.filter_depth = filter_depth
        self.input_shape = input_shape
        self.num_filters = num_filters

        # Size of outputs and filters

        self.filter_shape = (filter_depth, filter_size, filter_size)  # (6x6x3)
        self.output_shape = (filter_depth, input_height - filter_size + 1, input_width - filter_size + 1)

        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input_data):
        self.input_data = input_data
        # Initialized the input value
        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            for j in range(self.filter_depth):
                output[i] = correlate2d(self.input_data[j], self.filters[i][j], mode="valid")
        #Applying Relu Activtion function
        output = np.maximum(output, 0)
        return output

    def backward(self, dL_dout, lr):
        # Create a random dL_dout array to accommodate output gradients
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
                # Calculating the gradient of loss with respect to kernels
                dL_dfilters[i] = correlate2d(self.input_data, dL_dout[i],mode="valid")

                # Calculating the gradient of loss with respect to inputs
                dL_dinput += correlate2d(dL_dout[i],self.filters[i], mode="full")

        # Updating the parameters with learning rate
        self.filters -= lr * dL_dfilters
        self.biases -= lr * dL_dout

        # returning the gradient of inputs
        return dL_dinput

conv = Convolution(X_train[0].shape, 6, 3, 1)
print(conv.forward(X_train[1]))
