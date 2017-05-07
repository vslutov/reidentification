#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load model - minimal interface."""

from warnings import warn

import numpy as np
from skimage.transform import resize
from skimage.io import imread

from .models import models, ModelType
from .datasets import datasets, DatasetType

class ReidentificationFeatureBuilder(object):
    def __init__(self, model=ModelType.vgg16, dataset=DatasetType.market1501):
        """Init feature bulder."""
        dataset = datasets[dataset].get()
        model = models[model].get(epochs=10,
                                  X_train=dataset['X_train'],
                                  y_train=dataset['y_train'],
                                  triplets=False,
                                 )
        self.input_shape = dataset['X_train'].shape[2:]
        self.indexator = model.get_indexator()

    def generate_reidentification_features(self, image, bboxes):
        """Generate reidentification features."""
        X_test = []
        for bbox in bboxes:
            bbox = image[bbox["top"]:bbox["top"] + bbox["height"], bbox["left"]:bbox["left"] + bbox["width"]]
            bbox = resize(bbox, self.input_shape).transpose((2, 0, 1))
            X_test.append(bbox)
        return self.indexator.predict(np.array(X_test))

    def free(self):
        """Free resources."""
        warn('Not implemented yet')

def _test():
    image = imread("sample.jpg")
    feature_builder = ReidentificationFeatureBuilder()
    bboxes = [
      {"left": 10 + 10 * x, "top": 10 + 10 * x, "width": 10, "height": 10}
      for x in range(10)
    ]
    reidentification_features = feature_builder.generate_reidentification_features(image, bboxes)
    feature_builder.free()
