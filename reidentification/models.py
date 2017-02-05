#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neuronet reidentification model.
"""

import os.path
from functools import wraps

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras import backend as K

from .datasets import get_filepath

SIMPLE_H5 = get_filepath('simple.h5')

K.set_image_dim_ordering('tf')

def prepare_model(filename):
    """Save model after fit."""
    def decorator(func):
        """Decorator to prepare model."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper to prepare model."""
            model = func(*args, **kwargs)
            model.save(filepath)
            return model
        return wrapper
    return decorator

def get_model(filename, prepare_model):
    """Load or create model template"""
    def get_model_template(*args, **kwargs):
        """Load or create model template"""
        if not os.path.isfile(filename):
            print(_("{filename} not found... creating").format(filename=filename))
            return prepare_model(*args, **kwargs)
        else:
            return load_model(filename)

    return get_model_template

@prepare_model(SIMPLE_H5)
def prepare_simple(nb_epoch):
    """Prepare and safe simple model."""
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(128, 64, 3)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1502, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    (X_train, Y_train), (X_test, Y_test) = get_market1501()
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=nb_epoch, verbose=1)
    return model

get_simple = get_model(filename=SIMPLE_H5, prepare_model=prepare_simple)
