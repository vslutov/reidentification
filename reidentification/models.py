#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neuronet reidentification model.
"""

import os.path
from functools import wraps
from enum import Enum

from keras.models import Model, Sequential, load_model
from keras.applications import VGG16 as BaseVGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras import backend as K
from tabulate import tabulate

from .datasets import get_filepath, datasets, DatasetType
from .i18n import _

IMAGE_SHIFT = 0.2

K.set_image_dim_ordering('tf')

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def save_model(func):
    """Save model after fit."""
    @wraps(func)
    def wrapper(cls, nb_epoch, *args, **kwargs):
        """Wrapper to prepare model."""
        cls.model = func(cls, nb_epoch=nb_epoch, *args, **kwargs)
        cls.model.summary()

        market1501 = datasets[DatasetType.market1501].get()
        X_train = market1501['X_train']
        Y_train = market1501['Y_train']
        X_test = market1501['X_test']
        Y_test = market1501['Y_test']

        datagen = ImageDataGenerator(width_shift_range=IMAGE_SHIFT,
                                     height_shift_range=IMAGE_SHIFT,
                                     vertical_flip=True
                                    )
        datagen.fit(X_train)

        for i in range(nb_epoch):
            print("Epoch", i + 1, "/", nb_epoch)
            cls.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                    samples_per_epoch=len(X_train), nb_epoch=1, verbose=1)
            print(tabulate([cls.model.evaluate(X_test, Y_test, verbose=1)],
                           headers=cls.model.metrics_names))
        cls.model.save(cls.filepath)
        return cls.model
    return wrapper

class ReidModel:

    """Abstract model class"""

    filepath = None
    model = None

    @classmethod
    def get(cls, *args, **kwargs):
        """Get model."""
        if cls.model is None:
            if os.path.isfile(cls.filepath):
                cls.model = load_model(cls.filepath)
            else:
                print(_("{filepath} not found... creating").format(filepath=cls.filepath))
                cls.model = cls.prepare(*args, **kwargs)
        return cls.model

    @classmethod
    def prepare(cls, *args, **kwargs):
        """Should be replaced in successor."""
        raise NotImplementedError()

class Simple(ReidModel):

    """Simple model"""

    filepath = get_filepath('simple.h5')

    @classmethod
    @save_model
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

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return model

class VGG16(ReidModel):

    """VGG16 model"""

    filepath = get_filepath('vgg16.h5')

    @classmethod
    @save_model
    def prepare(cls, nb_epoch):
        """Prepare VGG model."""
        base_model = BaseVGG16(include_top=True)
        base_model.layers.pop()
        for layer in base_model.layers:
            layer.trainable = False
        top = base_model.layers[-1].output
        top = Dropout(0.5)(top)
        top = Dense(1502, activation='softmax')(top)
        model = Model(base_model.input, top)

        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return model

class ModelType(Enum):

    """Model type"""

    simple = 'simple'
    vgg16 = 'vgg16'

models = {ModelType.simple: Simple,
          ModelType.vgg16: VGG16
         }
