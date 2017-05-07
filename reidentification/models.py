
"""Neuronet reidentification model.
"""

import os.path
from functools import wraps
from enum import Enum
from pprint import pprint

import numpy as np
from keras.models import Model, Sequential, load_model
from keras.applications import VGG16 as BaseVGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Nadam
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from sklearn import neighbors, utils as sklearn_utils
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
from tabulate import tabulate

from .datasets import get_filepath, datasets, DatasetType
from .i18n import _

IMAGE_SHIFT = 0.2
IMAGE_ROTATION = 20
IMAGE_ZOOM = 0.2
BATCH_SIZE = 32

HASH_SIZE = 128

K.set_image_data_format('channels_last')

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

    FILENAME = None
    singleton = None

    @classmethod
    def get(cls, *args, **kwds):
        """Get singleton."""

        if cls.singleton is None:
            if cls.FILENAME is not None:
                filepath = get_filepath(cls.FILENAME)
                prefix, ext = os.path.splitext(filepath)
                if os.path.isfile(filepath):
                    if ext == '.h5':
                        model = load_model(filepath)
                    elif ext == '.pkl':
                        model = joblib.load(filepath)
                    else:
                        raise NotImplementedError()
                    cls.singleton = cls(model=model)
                else:
                    print(_("{filepath} not found... creating").format(filepath=filepath))

            if cls.singleton is None: # yet
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
        if self.FILENAME is not None:
            filepath = get_filepath(self.FILENAME)
            ext = os.path.splitext(filepath)[1]
            if ext == '.h5':
                self.model.save(filepath)
            elif ext == '.pkl':
                joblib.dump(self.model, filepath, compress=9)


class NNClassificator(ReidModel):

    head_size = None

    def fit(self, epochs, X_train, y_train):
        """Save model after fit."""
        self.model.summary()

        datagen = ImageDataGenerator(# width_shift_range=IMAGE_SHIFT,
                                     # height_shift_range=IMAGE_SHIFT,
                                     vertical_flip=True
                                    )
        datagen.fit(X_train)

        Y_train = np_utils.to_categorical(y_train)

        self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                 samples_per_epoch=len(X_train), epochs=epochs, verbose=1)

    def get_indexator(self):
        model = Model(self.model.input, self.model.layers[-1-self.head_size].output)
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        return model

    @classmethod
    def prepare(cls, epochs, X_train, y_train):
        model = cls(input_shape=X_train[0, :, :, :].shape, count=y_train.max() + 1)
        model.fit(X_train=X_train, y_train=y_train, epochs=epochs)
        model.save()
        return model

    def compile(self, lrm=0.002):
        self.model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.002 * lrm),
                           metrics=['accuracy'])

class LastClassifier(ReidModel):

    metric_names = ['crossentropy', 'accuracy', 'top 5 score']

    def __init__(self, indexator, model=None):
        self.indexator = indexator

    def index(self, X_test):
        return self.indexator.predict(X_test)

    def fit(self, X_test, y_test):
        X_feature = self.index(X_test)
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
        X_feature = self.index(X_query)
        return self.model.predict(X_feature)

    def _predict_proba(self, X_feature):
        return self.model.predict_proba(X_feature)

    def evaluate(self, X_query, y_query):
        X_feature = self.index(X_query)
        proba = self._predict_proba(X_feature)
        Y_query = np_utils.to_categorical(y_query)
        return (crossentropy_score((Y_query, proba)),
                accuracy_score((Y_query, proba)),
                top_5_score((Y_query, proba)),
               )

class FeatureDistance(LastClassifier):

    FILENAME = 'distance.pkl'

    N_NEIGHBORS = 5
    metric_names = ['rank-1', 'rank-{n_neighbors}'.format(n_neighbors=N_NEIGHBORS)]

    def __init__(self, indexator, model=None):
        super().__init__(indexator, model)
        self.y_test = None
        if not self.set_model(model):
            self.model = neighbors.NearestNeighbors(n_neighbors=self.N_NEIGHBORS)

    def fit(self, X_test, y_test):
        X_feature = self.index(X_test)
        self.model.fit(X_feature)
        self.y_test = y_test

    def predict(self, X_query):
        X_feature = self.index(X_query)
        neighbors = self.model.kneighbors(X_feature, n_neighbors=1, return_distance=False).reshape((-1,))
        return self.y_test(neighbors)

    def evaluate(self, X_query, y_query):
        X_feature = self.index(X_query)
        neighbors = self.model.kneighbors(X_feature, return_distance=False)

        y_pred = self.y_test[neighbors]
        y_true = np.hstack([np.array(y_query).reshape((-1, 1))] * self.N_NEIGHBORS)
        positive = y_pred == y_true
        return (positive[:, 1].sum() / y_query.size,
                positive.max(axis=1).sum() / y_query.size,
               )

class KNC(LastClassifier):

    FILENAME = 'knc.pkl'

    def __init__(self, indexator, model=None):
        super().__init__(indexator, model)
        if not self.set_model(model):
            self.model = neighbors.KNeighborsClassifier(n_neighbors=3)

class SVC(LastClassifier):

    FILENAME = 'svc.pkl'

    def __init__(self, indexator, model=None):
        super().__init__(indexator, model)
        if not self.set_model(model):
            self.model = LinearSVC()

    def _predict_proba(self, X_feature):
        return softmax(self.model.decision_function(X_feature))

class Simple(NNClassificator):

    """Simple model"""

    FILENAME = 'simple.h5'
    head_size = 2

    def __init__(self, input_shape=None, count=None, model=None):
        """Prepare simple model."""
        if not self.set_model(model):
            inputs = Input(shape=input_shape)

            top = Convolution2D(32, 3, 3, activation='relu')(inputs)
            top = Convolution2D(32, 3, 3, activation='relu')(top)
            top = MaxPooling2D(pool_size=(2,2))(top)
            top = Dropout(0.25)(top)
            top = Flatten()(top)
            top = Dense(128, activation='relu')(top)
            top = Dropout(0.5)(top)
            top = Dense(count, activation='softmax')(top)

            self.model = Model(inputs, top)
            self.compile()

class VGG16(NNClassificator):

    """VGG16 model"""

    FILENAME = 'vgg16.h5'
    head_size = 1

    def __init__(self, input_shape=None, count=None, model=None):
        """Prepare vgg16 model."""
        SLICE_LAYERS = 4

        if not self.set_model(model):
            base_model = BaseVGG16(include_top=False, input_shape=input_shape)
            for i in range(SLICE_LAYERS):
                pop_layer(base_model)
            for layer in base_model.layers:
                layer.trainable = False
            top = base_model.layers[-1].output
            top = GlobalAveragePooling2D()(top)
            top = BatchNormalization()(top)
            top = Dense(count, activation='softmax')(top)
            self.model = Model(base_model.input, top)
            self.compile()

    def unfreeze(self):
        for layer in self.model.layers:
            layer.trainable = True

    def fit(self, epochs, X_train, y_train):
        """Save model after fit."""
        self.model.summary()

        steps_per_epoch = len(X_train) // BATCH_SIZE
        datagen = ImageDataGenerator(# width_shift_range=IMAGE_SHIFT,
                                     # height_shift_range=IMAGE_SHIFT,
                                     vertical_flip=True
                                    )
        datagen.fit(X_train)

        X_train, y_train = sklearn_utils.shuffle(X_train, y_train)
        Y_train = np_utils.to_categorical(y_train)
        steps_per_epoch = int(0.9 * len(X_train) / BATCH_SIZE)
        X_train, X_val = X_train[:steps_per_epoch * BATCH_SIZE], X_train[steps_per_epoch * BATCH_SIZE:]
        Y_train, Y_val = Y_train[:steps_per_epoch * BATCH_SIZE], Y_train[steps_per_epoch * BATCH_SIZE:]
        validation_steps = len(X_val) // BATCH_SIZE

        print("First stage: learn top")
        self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                 steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                                 validation_data=(X_val, Y_val))

        self.unfreeze()
        self.compile(0.1)

        print("Second stage: fine-tune")
        self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                 steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                                 validation_data=(X_val, Y_val))

class ModelType(Enum):

    """Model type"""

    simple = 'simple'
    vgg16 = 'vgg16'

class ClassifierType(Enum):
    knc = 'knc'
    svc = 'svc'
    distance = 'distance'

models = {ModelType.simple: Simple,
          ModelType.vgg16: VGG16,
          ClassifierType.knc: KNC,
          ClassifierType.svc: SVC,
          ClassifierType.distance: FeatureDistance,
         }
