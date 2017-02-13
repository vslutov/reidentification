#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate model and print result.
"""

from tabulate import tabulate
import numpy as np
np.random.seed(123)

from .models import models, ModelType
from .datasets import datasets, DatasetType

def evaluate(args):
    """Evaluate model and print result."""
    dataset = datasets[DatasetType.market1501].get()
    if args.prepare:
        model = models[args.type].prepare(nb_epoch=args.nb_epoch,
                                          X_train=dataset['X_train'],
                                          y_train=dataset['y_train'],
                                         )
    else:
        model = models[args.type].get(nb_epoch=args.nb_epoch,
                                      X_train=dataset['X_train'],
                                      y_train=dataset['y_train'],
                                     )

    model = models[ModelType.final_classifier].prepare(indexator=model.get_indexator(),
                                                       X_test=dataset['X_test'],
                                                       y_test=dataset['y_test'],
                                                      )

    print(tabulate([model.evaluate(dataset['X_query'], dataset['y_query'])],
                   headers=model.metrics_names))
