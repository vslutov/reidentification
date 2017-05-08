#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate model and print result.
"""

from tabulate import tabulate
import numpy as np
np.random.seed(123)
from sklearn.decomposition import PCA
from sklearn import neighbors

from .models import models, ModelType, normalize_images
from .datasets import datasets, DatasetType

def evaluate(args):
    """Evaluate model and print result."""
    dataset = datasets[args.dataset].get()
    if args.prepare:
        model = models[args.model].prepare(epochs=args.epochs,
                                           X_train=dataset['X_train'],
                                           y_train=dataset['y_train'],
                                          )
    else:
        model = models[args.model].get(epochs=args.epochs,
                                       X_train=dataset['X_train'],
                                       y_train=dataset['y_train'],
                                      )

    N_NEIGHBORS = 5
    metric_names = ['rank-1', 'rank-{n_neighbors}'.format(n_neighbors=N_NEIGHBORS)]

    indexator = model.get_indexator()
    X_test = indexator.predict(normalize_images(dataset['X_test']))
    X_query = indexator.predict(normalize_images(dataset['X_query']))
    y_test=dataset['y_test']
    y_query=dataset['y_query']

    for i in range(16, 256, 8):
        pca = PCA(n_components=i)
        pca.fit(X_test)
        X_base = pca.transform(X_test) > 0
        X_find = pca.transform(X_query) > 0
        classifier = neighbors.NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='l1', n_jobs=-1)
        classifier.fit(X_base)
        our_neighbours = classifier.kneighbors(X_find, return_distance=False)
        y_pred = y_test[our_neighbours]
        y_true = np.hstack([np.array(y_query).reshape((-1, 1))] * N_NEIGHBORS)
        positive = y_pred == y_true
        precisions = (positive[:, 1].sum() / y_query.size,
                      positive.max(axis=1).sum() / y_query.size,
                     )
        with open('log.txt', 'a') as logfile:
            print(i, precisions, file=logfile)
        print(i, precisions)
