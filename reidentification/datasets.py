#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load datasets from REIDENTIFICATION_DATA_FOLDER environment."""

import os.path
import struct
import re
from zipfile import ZipFile

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from scipy import misc

from .i18n import _

REIDENTIFICATION_DATA_FOLDER = 'REIDENTIFICATION_DATA_FOLDER'
MARKET1501_NPZ = 'market1501.npz'
MARKET1501_ZIP = 'Market-1501-v15.09.15.zip'

def get_data_folder():
    """Get data folder from environment."""
    if REIDENTIFICATION_DATA_FOLDER in os.environ:
        return os.environ[REIDENTIFICATION_DATA_FOLDER]
    else:
        raise ValueError(_("Environment value {value} not found, please, set it to data folder")
                         .format(value=REIDENTIFICATION_DATA_FOLDER))

def get_mnist():
    """Get mnist dataset."""
    data_folder = get_data_folder()
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

def get_market1501():
    """Get market 1501 dataset."""
    data_folder = get_data_folder()
    npz_filename = os.path.join(data_folder, MARKET1501_NPZ)
    if not os.path.isfile(npz_filename):
        print(_("{npz_filename} not found... creating").format(npz_filename=npz_filename))
        prepare_market1501()
    npz_file = np.load(npz_filename)
    return (npz_file['X_train'], npz_file['Y_train']), (npz_file['X_test'], npz_file['Y_test'])

def prepare_market1501():
    """Unzip market1501 and save to npz."""
    def load_jpg(filepath, regexp, X, y):
        """Load jpg from archieve and save image and label."""
        match = regexp.search(filepath)
        if match:
            y.append(int(match.groups()[0]))
            with zip_file.open(match.string) as image_file:
                X.append(misc.imread(image_file).astype(np.float32) / 255)

    data_folder = get_data_folder()
    zip_filepath = os.path.join(data_folder, MARKET1501_ZIP)
    npz_filename = os.path.join(data_folder, MARKET1501_NPZ)

    if os.path.isfile(zip_filepath):

        X_train, y_train, X_test, y_test = [], [], [], []
        train_re = re.compile(r'^Market-1501-v15.09.15/bounding_box_train/' +
                              r'(\d{4})_c\ds\d_\d{6}_\d{2}.jpg$')
        test_re = re.compile(r'^Market-1501-v15.09.15/bounding_box_test/' +
                             r'(\d{4})_c\ds\d_\d{6}_\d{2}.jpg$')
        zip_file = ZipFile(zip_filepath)

        for filepath in zip_file.namelist():
            if train_re.search(filepath):
                load_jpg(filepath, train_re, X_train, y_train)
                load_jpg(filepath, test_re, X_test, y_test)

        Y_train = np_utils.to_categorical(y_train, 1502)
        X_train = np.array(X_train)
        Y_test = np_utils.to_categorical(y_test, 1502)
        X_test = np.array(X_test)
        np.savez_compressed(npz_filename, X_train=X_train, Y_train=Y_train,
                            X_test=X_test, Y_test=Y_test)
    else:
        raise ValueError(_("{filepath} not found, please, download it from {url}")
                         .format(filepath=zip_filepath,
                                 url='https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view'))
