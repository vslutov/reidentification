#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate model and print result.
"""

import numpy as np
np.random.seed(123)

from .model import model
from .datasets import get_mnist

def evaluate(args):
    """Evaluate model and print result."""
    (X_train, Y_train), (X_test, Y_test) = get_mnist()
    model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
    print(model.evaluate(X_test, Y_test, verbose=0))
