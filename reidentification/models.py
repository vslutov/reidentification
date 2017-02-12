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
from sklearn.neighbors import KNeighborsClassifier

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

class ReidModel:

    """Abstract model class"""

    filepath = None
    model = None

    # @classmethod
    # def get(cls, *args, **kwargs):
    #     """Get model."""
    #     if cls.model is None:
    #         if os.path.isfile(cls.filepath):
    #             cls.model = load_model(cls.filepath)
    #         else:
    #             print(_("{filepath} not found... creating").format(filepath=cls.filepath))
    #             cls.model = cls.prepare(*args, **kwargs)
    #     return cls.model

    @classmethod
    def __init__(cls, *args, **kwargs):
        """Should be replaced in successor."""
        raise NotImplementedError()

    def save(self):
        """Save model after fit."""
        self.model.save(self.filepath)


class NNClassificator(ReidModel):

    def fit(self, X_train, Y_train):
        """Save model after fit."""
        self.model.summary()

        datagen = ImageDataGenerator(width_shift_range=IMAGE_SHIFT,
                                     height_shift_range=IMAGE_SHIFT,
                                     vertical_flip=True
                                    )
        datagen.fit(X_train)

        for i in range(nb_epoch):
            print("Epoch", i + 1, "/", nb_epoch)
            self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                     samples_per_epoch=len(X_train), nb_epoch=1, verbose=1)
            print(tabulate([self.model.evaluate(X_test, Y_test, verbose=1)],
                            headers=self.model.metrics_names))

    def get_indexator(self):
        model = Model(self.model.input, self.model.layers[-2].output)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return model

    @classmethod
    def prepare(cls, nb_epoch):
        model = cls()
        market1501 = datasets[DatasetType.market1501].get()
        model.fit(market1501['X_train'], market1501['Y_train'])
        model.save()

class FinalClassificator(ReidModel):

    filename = get_filepath("classificator.h5")

    def __init__(self, indexator):
        self.indexator = indexator

    def fit(self, X_test, y_test):
        X_feature = self.indexator.predict(X_test)
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X_feature, y_test)

    def predict(self, X_query):
        X_query = self.indexator.predict(X_query)
        return self.model.predict_proba(X_query)

class Simple(NNClassificator):

    """Simple model"""

    filepath = get_filepath('simple.h5')

    def __init__(self):
        """Prepare simple model."""
        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(128, 64, 3)))
        self.model.add(Convolution2D(32, 3, 3, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1502, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy', 'top_k_categorical_accuracy'])

class VGG16(ReidModel):

    """VGG16 model"""

    filepath = get_filepath('vgg16.h5')

    # @classmethod
    # @save_model
    # def prepare(cls, nb_epoch):
    #     """Prepare VGG model."""
    #     base_model = BaseVGG16(include_top=True)
    #     base_model.layers.pop()
    #     for layer in base_model.layers:
    #         layer.trainable = False
    #     top = base_model.layers[-1].output
    #     top = Dropout(0.5)(top)
    #     top = Dense(1502, activation='softmax')(top)
    #     model = Model(base_model.input, top)

    #     model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    #     return model

class ModelType(Enum):

    """Model type"""

    simple = 'simple'
    vgg16 = 'vgg16'

models = {ModelType.simple: Simple,
          ModelType.vgg16: VGG16
         }
