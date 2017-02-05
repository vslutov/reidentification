#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate model and print result.
"""

from tabulate import tabulate
import numpy as np
np.random.seed(123)

from .models import get_simple
from .datasets import get_market1501

def evaluate(args):
    """Evaluate model and print result."""
    (X_train, Y_train), (X_test, Y_test) = get_market1501()
    model = get_simple(nb_epoch=args.nb_epoch)
    print(tabulate([model.evaluate(X_test, Y_test, verbose=0)],
                   headers=model.metrics_names))
