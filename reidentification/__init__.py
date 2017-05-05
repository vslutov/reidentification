#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reidentification experiments.

It uses keras, you can use theano and cuDNN for fast learning.
"""
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
