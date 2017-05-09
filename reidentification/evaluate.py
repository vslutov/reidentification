#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate model and print result.
"""

from tabulate import tabulate
import numpy as np
from sklearn.decomposition import PCA
from sklearn import neighbors

from .models import models, ModelType, normalize_images
from .datasets import datasets, DatasetType

def evaluate(args):
    """Evaluate model, pca or adaboost."""
    if args.pca:
        return evaluate_pca(args)
    else:
        return evaluate_normal(args)

def get_dataset_and_model(args):
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

    classifier = models[args.classifier].prepare(indexator=model.get_indexator(),
                                                 X_test=dataset['X_test'],
                                                 y_test=dataset['y_test'],
                                                )
    return dataset, model

def evaluate_normal(args):
    """Evaluate model and print result."""
    np.random.seed(123)
    dataset, model = get_dataset_and_model(args)

    print(tabulate([classifier.evaluate(dataset['X_query'], dataset['y_query'])],
                   headers=classifier.metric_names))

def evaluate_pca(args):
    """Evaluate pca and print result."""
    dataset, model = get_dataset_and_model(args)

    N_NEIGHBORS = 5
    metric_names = ['rank-1', 'rank-{n_neighbors}'.format(n_neighbors=N_NEIGHBORS)]

    indexator = model.get_indexator()
    X_test = indexator.predict(normalize_images(dataset['X_test']))
    X_query = indexator.predict(normalize_images(dataset['X_query']))
    y_test=dataset['y_test']
    y_query=dataset['y_query']

    X_train = indexator.predict(normalize_images(dataset['X_train']))
    pca = PCA(n_components=512)
    pca.fit(X_train)
    _X_base = (pca.transform(X_test) > 0)
    _X_find = (pca.transform(X_query) > 0)

    for i in range(16, 513, 4):
        X_base = _X_base[:, :i]
        X_find = _X_find[:, :i]
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

