#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neuronet reidentification model.
"""

import os.path
from functools import wraps
from enum import Enum

from keras.models import Sequential, load_model
from keras.applications import VGG16 as BaseVGG16
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras import backend as K

from .datasets import get_filepath, get_market1501
from .i18n import _


class ModelType(Enum):

    """Model type"""

    simple = 'simple'
    vgg16 = 'vgg16'

K.set_image_dim_ordering('tf')

def prepare_model(func):
    """Save model after fit."""
    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        """Wrapper to prepare model."""
        model = func(cls, *args, **kwargs)
        model.save(cls.filepath)
        return model
    return wrapper

class Model:

    """Abstract class for model"""

    @classmethod
    def get(cls, *args, **kwargs):
        """Get model."""
        if not os.path.isfile(cls.filepath):
            print(_("{filepath} not found... creating").format(filepath=cls.filepath))
            return cls.prepare(*args, **kwargs)
        else:
            return load_model(cls.filepath)

    @classmethod
    def prepare(cls, *args, **kwargs):
        """Should be replaced in successor."""
        raise NotImplementedError()

class Simple(Model):

    """Simple model"""

    filepath = get_filepath('simple.h5')

    @classmethod
    @prepare_model
    def prepare(cls, nb_epoch):
        """Prepare simple model."""
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

class VGG16(Model):

    """VGG16 model"""

    filepath = get_filepath('vgg16.h5')

    @classmethod
    @prepare_model
    def prepare(cls, nb_epoch):
        """Prepare VGG model."""
        model = BaseVGG16(include_top=False, input_shape=(128, 64, 3))
        model.add(Dence(4096, activation='relu'))
        model.add(Dence(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1502, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        (X_train, Y_train), (X_test, Y_test) = get_market1501()
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=nb_epoch, verbose=1)
        return model

models = {ModelType.simple: Simple,
          ModelType.vgg16: VGG16
         }
