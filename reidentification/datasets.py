#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load datasets from REIDENTIFICATION_DATA_FOLDER environment."""

import os.path
import struct
import re
from zipfile import ZipFile
from enum import Enum
from functools import wraps
import random

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from skimage.transform import resize
from skimage.io import imread
from hdf5storage import loadmat

from .i18n import _

REIDENTIFICATION_DATA_FOLDER = 'REIDENTIFICATION_DATA_FOLDER'

def get_data_folder() -> str:
    """Get data folder from environment."""
    if REIDENTIFICATION_DATA_FOLDER in os.environ:
        return os.environ[REIDENTIFICATION_DATA_FOLDER]
    else:
        raise ValueError(_("Environment value {value} not found, please, set it to data folder")
                         .format(value=REIDENTIFICATION_DATA_FOLDER))

def preprocess_cathegories(y):
    vals = sorted(list(set(y)))
    return np.array([vals.index(x) for x in y], dtype=np.int32)

def get_filepath(filename: str, *args) -> str:
    """Get filepath for file from data_dir."""
    return os.path.join(get_data_folder(), filename, *args)

MARKET1501_ZIP = get_filepath('Market-1501-v15.09.15.zip')
CUHK03_MAT = get_filepath('cuhk-03.mat')
CUHK03_OUTPUT_SIZE = (128, 64)

def save_dataset(func):
    """Save dataset after extract."""
    @wraps(func)
    def wrapper(cls, *args, **kwargs):
        """Wrapper to prepare model."""
        cls.dataset = func(cls, *args, **kwargs)
        np.savez_compressed(cls.filepath, **cls.dataset)
        return cls.dataset
    return wrapper

class Dataset:

    """Abstract dataset class"""

    filepath = None
    dataset = None

    @classmethod
    def get(cls, *args, **kwargs):
        """Get dataset."""
        if cls.dataset is None:
            if os.path.isfile(cls.filepath):
                cls.dataset = np.load(cls.filepath)
            else:
                print(_("{filepath} not found... creating").format(filepath=cls.filepath))
                cls.dataset = cls.prepare(*args, **kwargs)
        return cls.dataset

    @classmethod
    def prepare(cls, *args, **kwargs):
        """Should be replaced in successor."""
        raise NotImplementedError()

class MNIST(Dataset):

    """MNIST dataset"""

    filepath = get_filepath('mnist.npz')

    @classmethod
    @save_dataset
    def prepare(cls):
        """Prepare dataset."""
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_train /= 255
        X_test /= 255
        Y_train = np_utils.to_categorical(y_train, 10)
        Y_test = np_utils.to_categorical(y_test, 10)
        return {'X_train': X_train,
                'Y_train': Y_train,
                'X_test': X_test,
                'Y_test': Y_test}

class Market1501(Dataset):

    """Market1501 dataset"""

    filepath = get_filepath('market1501.npz')

    @classmethod
    @save_dataset
    def prepare(cls):
        """Unzip market1501 and save to npz."""
        def load_jpg(filepath, regexp, X, y):
            """Load jpg from archieve and save image and label."""
            match = regexp.search(filepath)
            if match:
                y.append(int(match.groups()[0]))
                with zip_file.open(match.string) as image_file:
                    image = imread(image_file).astype(np.float32) / 255
                    # image = resize(image, (224, 224))
                    X.append(image)

        if os.path.isfile(MARKET1501_ZIP):

            X_train, y_train, X_test, y_test, X_query, y_query = [], [], [], [], [], []
            train_re = re.compile(r'^Market-1501-v15.09.15/bounding_box_train/' +
                                  r'(-?\d+)_c\ds\d_\d{6}_\d{2}.jpg[.\w]*$')
            test_re = re.compile(r'^Market-1501-v15.09.15/bounding_box_test/' +
                                 r'(-?\d+)_c\ds\d_\d{6}_\d{2}.jpg[.\w]*$')
            query_re = re.compile(r'^Market-1501-v15.09.15/query/' +
                                  r'(-?\d+)_c\ds\d_\d{6}_\d{2}.jpg[.\w]*$')

            with ZipFile(MARKET1501_ZIP) as zip_file:
                for filepath in zip_file.namelist():
                    load_jpg(filepath, train_re, X_train, y_train)
                    load_jpg(filepath, test_re, X_test, y_test)
                    load_jpg(filepath, query_re, X_query, y_query)

            X_train = np.array(X_train)
            y_train = preprocess_cathegories(y_train)

            X_test = np.array(X_test)
            y_test = preprocess_cathegories(y_test)

            X_query = np.array(X_query)
            y_query = preprocess_cathegories(y_query) + 2

            return {'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'X_query': X_query,
                    'y_query': y_query,
                   }
        else:
            raise ValueError(_("{filepath} not found, please, download it from {url}")
                             .format(filepath=MARKET1501_ZIP,
                                     url='https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'))

class CUHK03(Dataset):

    """CUHK03 dataset"""

    filepath = get_filepath('cuhk03.npz')

    @classmethod
    @save_dataset
    def prepare(cls):
        """Unzip market1501 and save to npz."""
        if os.path.isfile(CUHK03_MAT):
            cuhk03 = loadmat(CUHK03_MAT)
            labeled = cuhk03['labeled']
            detected = cuhk03['detected']

            X_train, y_train, X_test, y_test, X_query, y_query = [], [], [], [], [], []
            ident_count = 0
            for cam in range(5):
                for ident in range(labeled[cam, 0].shape[0]):
                    images = list(labeled[cam, 0][ident]) + list(detected[cam, 0][ident])
                    images = [resize(im, CUHK03_OUTPUT_SIZE)
                              for im in images if len(im) != 0]
                    random.shuffle(images)
                    if random.uniform(0, 1) > 0.5:
                        X_train.extend(images)
                        y_train.extend([ident_count] * len(images))
                    else:
                        X_query.append(images[0])
                        X_test.extend(images[1:])
                        y_query.append(ident_count)
                        y_test.extend([ident_count] * (len(images) - 1))
                    ident_count += 1
            X_train = np.array(X_train)
            y_train = preprocess_cathegories(y_train)
            X_test = np.array(X_test)
            y_test = preprocess_cathegories(y_test)
            X_query = np.array(X_query)
            y_query = preprocess_cathegories(y_query)
            return {'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'X_query': X_query,
                    'y_query': y_query,
                   }
        else:
            raise ValueError(_("{filepath} not found, please, download it from {url}")
                             .format(filepath=CUHK03_MAT,
                                     url='http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html'))

class DatasetType(Enum):

    """Model type"""

    mnist = 'mnist'
    market1501 = 'market1501'
    cuhk03 = 'cuhk03'

datasets = {DatasetType.mnist: MNIST,
            DatasetType.market1501: Market1501,
            DatasetType.cuhk03: CUHK03,
           }
