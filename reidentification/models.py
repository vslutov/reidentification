#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neuronet reidentification model.
"""

import os.path
from functools import wraps
from enum import Enum

import numpy as np
from keras.models import Model, Sequential, load_model
from keras.applications import VGG16 as BaseVGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
from tabulate import tabulate

from .datasets import get_filepath, datasets, DatasetType
from .i18n import _

IMAGE_SHIFT = 0.2

K.set_image_dim_ordering('th')

def compile_score(func):
    Y_query = K.placeholder(shape=(None, None))
    proba = K.placeholder(shape=(None, None))
    return K.function([Y_query, proba], func(Y_query, proba))

crossentropy_score = compile_score(metrics.categorical_crossentropy)
accuracy_score = compile_score(metrics.categorical_accuracy)
top_5_score = compile_score(metrics.top_k_categorical_accuracy)

def softmax(decision_function):
    exp = np.exp(decision_function)
    return exp / exp.sum(axis=1).reshape((-1, 1))

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
    singleton = None

    @classmethod
    def get(cls, *args, **kwds):
        """Get singleton."""
        ext = os.path.splitext(cls.filepath)[1]

        if cls.singleton is None:
            if os.path.isfile(cls.filepath):
                if ext == '.h5':
                    model = load_model(cls.filepath)
                elif ext == '.pkl':
                    model = joblib.load(cls.filepath)
                else:
                    raise NotImplementedError()
                cls.singleton = cls(model=model)
            else:
                print(_("{filepath} not found... creating").format(filepath=cls.filepath))
                cls.singleton = cls.prepare(*args, **kwds)
        return cls.singleton

    def set_model(self, model):
        if model is not None:
            self.model = model
            return True
        else:
            return False

    @classmethod
    def __init__(cls, *args, **kwds):
        """Should be replaced in successor."""
        raise NotImplementedError()

    def prepare(self, *args, **kwds):
        raise NotImplementedError()

    def save(self):
        """Save singleton after fit."""
        ext = os.path.splitext(self.filepath)[1]
        if ext == '.h5':
            self.model.save(self.filepath)
        elif ext == '.pkl':
            joblib.dump(self.model, self.filepath, compress=9)


class NNClassificator(ReidModel):

    head_size = None

    def fit(self, nb_epoch, X_train, y_train):
        """Save model after fit."""
        self.model.summary()

        datagen = ImageDataGenerator(# width_shift_range=IMAGE_SHIFT,
                                     # height_shift_range=IMAGE_SHIFT,
                                     vertical_flip=True
                                    )
        datagen.fit(X_train)

        Y_train = np_utils.to_categorical(y_train)

        self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                 samples_per_epoch=len(X_train), nb_epoch=nb_epoch, verbose=1)

    def get_indexator(self):
        model = Model(self.model.input, self.model.layers[-1-self.head_size].output)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return model

    @classmethod
    def prepare(cls, nb_epoch, X_train, y_train):
        model = cls(input_shape=X_train[0, :, :, :].shape, count=y_train.max() + 1)
        model.fit(X_train=X_train, y_train=y_train, nb_epoch=nb_epoch)
        model.save()
        return model

class LastClassifier(ReidModel):

    metrics_names = ['crossentropy', 'accuracy', 'top 5 score']

    def __init__(self, indexator, model=None):
        self.indexator = indexator

    def fit(self, X_test, y_test):
        X_feature = self.indexator.predict(X_test)
        self.model.fit(X_feature, y_test)

    @classmethod
    def prepare(cls, indexator, X_test, y_test):
        model = cls(indexator)
        model.fit(X_test=X_test,
                  y_test=y_test,
                 )
        model.save()
        return model

    def predict(self, X_query):
        X_feature = self.indexator.predict(X_query)
        return self.model.predict(X_feature)

    def predict_proba(self, X_feature):
        return self.model.predict_proba(X_feature)

    def evaluate(self, X_query, y_query):
        X_feature = self.indexator.predict(X_query)
        proba = self.predict_proba(X_feature)
        Y_query = np_utils.to_categorical(y_query)
        return [crossentropy_score((Y_query, proba)),
                accuracy_score((Y_query, proba)),
                top_5_score((Y_query, proba))
               ]

class KNC(LastClassifier):

    filepath = get_filepath('knc.pkl')

    def __init__(self, indexator, model=None):
        super().__init__(indexator, model)
        if not self.set_model(model):
            self.model = KNeighborsClassifier(n_neighbors=3)

class SVC(LastClassifier):

    filepath = get_filepath('svc.pkl')

    def __init__(self, indexator, model=None):
        super().__init__(indexator, model)
        if not self.set_model(model):
            self.model = LinearSVC()

    def predict_proba(self, X_feature):
        return softmax(self.model.decision_function(X_feature))

class Simple(NNClassificator):

    """Simple model"""

    filepath = get_filepath('simple.h5')
    head_size = 2

    def __init__(self, input_shape=None, count=None, model=None):
        """Prepare simple model."""
        if not self.set_model(model):
            self.model = Sequential()

            self.model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=input_shape))
            self.model.add(Convolution2D(32, 3, 3, activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(count, activation='softmax'))

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

class ClassifierType(Enum):
    knc = 'knc'
    svc = 'svc'

models = {ModelType.simple: Simple,
          ModelType.vgg16: VGG16,
          ClassifierType.knc: KNC,
          ClassifierType.svc: SVC,
         }
