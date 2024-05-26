"""Module that define the network."""

# Import standard library
from typing import List, Union 

# Import third-parties library
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


class ClassifierWood(tf.keras.Model):
    """Class which define the model.

    Args:
        tf (_type_): _description_
    """

    def __init__(self, filters: List[int], kernel_size: Union[int, List[int]])