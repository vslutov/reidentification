#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load datasets from REIDENTIFICATION_DATA_FOLDER environment."""

import os.path
import struct
import re
from zipfile import ZipFile
from enum import Enum
from functools import wraps

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from scipy import misc

from .i18n import _

REIDENTIFICATION_DATA_FOLDER = 'REIDENTIFICATION_DATA_FOLDER'

def get_data_folder() -> str:
    """Get data folder from environment."""
    if REIDENTIFICATION_DATA_FOLDER in os.environ:
        return os.environ[REIDENTIFICATION_DATA_FOLDER]
    else:
        raise ValueError(_("Environment value {value} not found, please, set it to data folder")
                         .format(value=REIDENTIFICATION_DATA_FOLDER))

def get_filepath(filename: str, *args) -> str:
    """Get filepath for file from data_dir."""
    return os.path.join(get_data_folder(), filename, *args)

MARKET1501_ZIP = get_filepath('Market-1501-v15.09.15.zip')

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
                cls.dataset = cls.prepare(cls, *args, **kwargs)
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
                    X.append(misc.imread(image_file).astype(np.float32) / 255)

        if os.path.isfile(MARKET1501_ZIP):

            X_train, y_train, X_test, y_test = [], [], [], []
            train_re = re.compile(r'^Market-1501-v15.09.15/bounding_box_train/' +
                                  r'(\d{4})_c\ds\d_\d{6}_\d{2}.jpg$')
            train_re2 = re.compile(r'^Market-1501-v15.09.15/bounding_box_test/' +
                                   r'(\d{4})_c\ds\d_\d{6}_\d{2}.jpg$')
            test_re = re.compile(r'^Market-1501-v15.09.15/query/' +
                                 r'(\d{4})_c\ds\d_\d{6}_\d{2}.jpg$')

            with ZipFile(MARKET1501_ZIP) as zip_file:
                for filepath in zip_file.namelist():
                    load_jpg(filepath, train_re, X_train, y_train)
                    load_jpg(filepath, train_re2, X_train, y_train)
                    load_jpg(filepath, test_re, X_test, y_test)

            Y_train = np_utils.to_categorical(y_train, 1502)
            X_train = np.array(X_train)
            Y_test = np_utils.to_categorical(y_test, 1502)
            X_test = np.array(X_test)
            return {'X_train': X_train,
                    'Y_train': Y_train,
                    'X_test': X_test,
                    'Y_test': Y_test}
        else:
            raise ValueError(_("{filepath} not found, please, download it from {url}")
                             .format(filepath=MARKET1501_ZIP,
                                     url='https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'))

class DatasetType(Enum):

    """Model type"""

    mnist = 'mnist'
    market1501 = 'market1501'

datasets = {DatasetType.mnist: MNIST,
            DatasetType.market1501: Market1501
           }
