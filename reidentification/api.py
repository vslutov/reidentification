#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load model - minimal interface."""

from warnings import warn

from scipy.misc import imresize

from .models import models, ModelType
from .datasets import datasets, DatasetType

class ReidentificationFeatureBuilder(object):
    def __init__(self, model=ModelType.VGG16, dataset=DatasetType.market1501):
        """Init feature bulder."""
        dataset = datasets[dataset].get()
        model = models[model].get(nb_epoch=10,
                                  X_train=dataset['X_train'],
                                  y_train=dataset['y_train'],
                                  triplets=False,
                                 )
        self.input_shape = dataset['X_train'].shape[2:]
        self.indexator = model.get_indexator()
        print(self.input_shape)

    def generate_reidentification_features(self, im, bboxes):
        """Generate reidentification features."""
        X_test = []
        for bbox in bboxes:
            bbox = im[bbox["right"]:bbox["right"] + bbox["height"], bbox["left"]:bbox["left"] + bbox["width"]]
            bbox = imresize(bbox, self.input_shape)
            print(bbox.shape)

    def free(slef):
        """Free resources."""
        warn('Not implemented yet')

def evaluate(args):
    """Evaluate model and print result."""
    if args.prepare:
        model = models[args.model].prepare(nb_epoch=args.nb_epoch,
                                           X_train=dataset['X_train'],
                                           y_train=dataset['y_train'],
                                           triplets=args.triplets,
                                          )
    else:
        model = models[args.model].get(nb_epoch=args.nb_epoch,
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
