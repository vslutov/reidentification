#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Neuronet reidentification model.
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras import backend as K

K.set_image_dim_ordering('tf')

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
