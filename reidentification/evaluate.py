#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate model and print result.
"""

from tabulate import tabulate
import numpy as np
np.random.seed(123)

from .models import models
from .datasets import datasets, DatasetType

def evaluate(args):
    """Evaluate model and print result."""
    market1501 = datasets[DatasetType.market1501].get()
    model = models[args.type].get(nb_epoch=args.nb_epoch)
    print(tabulate([model.evaluate(market1501['X_test'], market1501['Y_test'], verbose=0)],
                   headers=model.metrics_names))
