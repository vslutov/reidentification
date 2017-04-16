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
    dataset = datasets[args.dataset].get()
    if args.prepare:
        model = models[args.model].prepare(epochs=args.epochs,
                                           X_train=dataset['X_train'],
                                           y_train=dataset['y_train'],
                                           triplets=args.triplets,
                                          )
    else:
        model = models[args.model].get(epochs=args.epochs,
                                       X_train=dataset['X_train'],
                                       y_train=dataset['y_train'],
                                       triplets=args.triplets,
                                      )

    classifier = models[args.classifier].prepare(indexator=model.get_indexator(),
                                                 X_test=dataset['X_test'],
                                                 y_test=dataset['y_test'],
                                                )

    print(tabulate([classifier.evaluate(dataset['X_query'], dataset['y_query'])],
                   headers=classifier.metric_names))
