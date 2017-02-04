#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load datasets from REIDENTIFICATION_DATA_FOLDER environment."""

import os.path
import struct

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from .i18n import _

REIDENTIFICATION_DATA_FOLDER = 'REIDENTIFICATION_DATA_FOLDER'

def get_data_folder():
    """Get data folder from environment."""
    if REIDENTIFICATION_DATA_FOLDER in os.environ:
        return os.environ[REIDENTIFICATION_DATA_FOLDER]
    else:
        raise ValueError(_("Can't find environment value {value}")
                         .format(value=REIDENTIFICATION_DATA_FOLDER))

def get_mnist(dataset = "training"):
    """Get mnist dataset."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return (X_train, Y_train), (X_test, Y_test)
