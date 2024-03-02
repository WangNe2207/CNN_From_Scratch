import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from scipy.signal import correlate2d

(i_train, l_train), (i_test, l_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

i_train = i_train[0:20000]
l_train = l_train[0:20000]
i_valid = i_train[20000:30000]
l_valid = l_train[20000:30000]

l_train = to_categorical(l_train, len(class_names))
l_valid = to_categorical(l_valid, len(class_names))

def show_images(train_images,
                class_names,
                train_labels):
    plt.figure(figsize=(12, 12))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images, cmap=plt.colormaps['binary'])
    plt.xlabel(class_names[train_labels])
    plt.show()


# show_images(i_train[1], class_names, l_train[1])

class Convolution:

    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.num_filters = num_filters
        self.input_shape = input_shape

        # Size of outputs and filters

        self.filter_shape = (num_filters, filter_size, filter_size)  # (3,3)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)

        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)


