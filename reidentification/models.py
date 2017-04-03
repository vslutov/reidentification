
"""Neuronet reidentification model.
"""

import os.path
import random
from functools import wraps
from enum import Enum
from pprint import pprint

import numpy as np
from keras.models import Model, Sequential, load_model
from keras.applications import VGG16 as BaseVGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization, Input, TimeDistributed
from keras.optimizers import Nadam, SGD
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.utils.visualize_util import plot
from sklearn import neighbors
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
from tabulate import tabulate
from scipy.spatial.distance import squareform, pdist

from .datasets import get_filepath, datasets, DatasetType
from .i18n import _
from .triplets import triplet_loss

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
        if cls.singleton is None:
            if cls.filepath is not None:
                filepath = get_filepath(cls.filepath)
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
        if self.filepath is not None:
            filepath = get_filepath(self.filepath)
            ext = os.path.splitext(filepath)[1]
            if ext == '.h5':
                self.model.save(filepath)
            elif ext == '.pkl':
                joblib.dump(self.model, filepath, compress=9)


class NNClassifier(ReidModel):

    head_size = None

    def fit(self, nb_epoch, X_train, y_train, triplets):
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
    def prepare(cls, nb_epoch, X_train, y_train, triplets):
        model = cls(input_shape=X_train[0, :, :, :].shape, count=y_train.max() + 1)
        model.fit(X_train=X_train, y_train=y_train, nb_epoch=nb_epoch, triplets=triplets)
        model.save()
        return model

    def compile(self, lrm=0.002, loss=None):
        optimizer = Nadam(lr=0.01 * lrm)
        if loss is None:
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                               metrics=['accuracy'])
        else:
            self.model.compile(loss=loss, optimizer=optimizer,
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

    filepath = 'distance.pkl'

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

    filepath = 'knc.pkl'

    def __init__(self, indexator, model=None):
        super().__init__(indexator, model)
        if not self.set_model(model):
            self.model = neighbors.KNeighborsClassifier(n_neighbors=3)

class SVC(LastClassifier):

    filepath = 'svc.pkl'

    def __init__(self, indexator, model=None):
        super().__init__(indexator, model)
        if not self.set_model(model):
            self.model = LinearSVC()

    def _predict_proba(self, X_feature):
        return softmax(self.model.decision_function(X_feature))

class Simple(NNClassifier):

    """Simple model"""

    filepath = 'simple.h5'
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

class VGG16(NNClassifier):

    """VGG16 model"""

    filepath = 'vgg16.h5'
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
            # top = Dropout(0.5)(top)
            top = Dense(count, activation='softmax')(top)
            self.model = Model(base_model.input, top)
            self.compile()

    def unfreeze(self):
        for layer in self.model.layers:
            layer.trainable = True

    def fit(self, nb_epoch, X_train, y_train, triplets=False):
        """Save model after fit."""
        self.model.summary()
        if triplets:
            self.unfreeze()
            self.compile()
            triplet_model = TripletLossOptimizer(base_model=self.get_indexator())
            triplet_model.fit(nb_epoch=nb_epoch, X_train=X_train, y_train=y_train)
        else:
            datagen = ImageDataGenerator(# width_shift_range=IMAGE_SHIFT,
                                         # height_shift_range=IMAGE_SHIFT,
                                         vertical_flip=True
                                        )
            datagen.fit(X_train)

            Y_train = np_utils.to_categorical(y_train)

            print("First stage: learn top")
            self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                     samples_per_epoch=len(X_train), nb_epoch=nb_epoch, verbose=1)

            self.unfreeze()
            self.compile(0.1)

            print("Second stage: fine-tune")
            self.model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                     samples_per_epoch=len(X_train), nb_epoch=nb_epoch, verbose=1)

class TripletLossOptimizer(NNClassifier):

    def get_triplets(self, X_train, y_train, batch_size):
        while True:
            X_feature = self.base_model.predict(X_train)
            feature_size = X_feature[0].size
            D2 = squareform(pdist(X_feature))
            result = []
            for i, k in enumerate(y_train):
                mask = np.zeros(y_train.size, dtype=bool)
                mask[i] = True
                group = np.where(np.logical_and(y_train == k, np.logical_not(mask)))[0]
                num_in_group = np.argmin(D2[i, group])
                others = np.where(np.logical_and(y_train != k, D2[i, :] > D2[i, group[num_in_group]]))[0]
                if len(others) == 0:
                    continue

                num_in_others = np.argmin(D2[i, others])

                anchor = X_train[i]
                positive = X_train[group[num_in_group]]
                negative = X_train[others[num_in_others]]

                result.append([anchor, positive, negative])

            random.shuffle(result)

            while len(result) > batch_size:
                yield (np.array(result[:batch_size]), np.zeros((batch_size, 3, feature_size)))
                result = result[batch_size:]

    def __init__(self, base_model=None, model=None):
        if not self.set_model(model):
            self.base_model = base_model
            input_triplets = Input(shape=(3, ) + base_model.input_shape[1:])
            top = TimeDistributed(base_model)(input_triplets)
            self.model = Model(input_triplets, top)
            self.compile(loss=triplet_loss)

    def fit(self, nb_epoch, X_train, y_train):
        self.model.fit_generator(self.get_triplets(X_train, y_train, batch_size=32),
                                 samples_per_epoch=len(X_train), nb_epoch=nb_epoch, verbose=1)

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
