#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reidentification experiments.

Main module - parse cli and run a functions.
"""

import argparse
from enum import Enum

from .i18n import _
from .evaluate import evaluate as run_evaluate
from .datasets import datasets, DatasetType
from .models import models, ModelType, ClassifierType

def prepare_dataset(args):
    """Prepare model."""
    print(_("Preparing {type}").format(type=args.type))
    datasets[args.type].prepare()

def prepare_model(args):
    """Prepare dataset."""
    print(_("Preparing {type}").format(type=args.type))
    dataset = datasets[args.dataset].get()
    models[args.type].prepare(nb_epoch=args.nb_epoch,
                              X_train=dataset['X_train'],
                              y_train=dataset['y_train'],
                              triplets=args.triplets,
                             )

def main():
    """Main function - parse command-line arguments and run function."""
    parser = argparse.ArgumentParser(description=_("Run reidentification experiments."))

    subparsers = parser.add_subparsers(title='actions', help=_("system actions"))

    evaluate = subparsers.add_parser('evaluate', help=_("evaluate model"))
    evaluate.add_argument('model', choices=ModelType, type=ModelType, help=_("model type"))
    evaluate.add_argument('-c', '--classifier', type=ClassifierType, help=_("classifier for model"), default=ClassifierType.distance)
    evaluate.add_argument('-d', '--dataset', type=DatasetType, help=_("dataset"), default=DatasetType.market1501)
    evaluate.add_argument('-e', '--nb_epoch', type=int, default=10, help=_("epoch count"))
    evaluate.add_argument('--prepare', dest='prepare', action='store_true')
    evaluate.set_defaults(prepare=False)
    evaluate.add_argument('--triplets', dest='triplets', action='store_true')
    evaluate.set_defaults(triplets=False)
    evaluate.set_defaults(func=run_evaluate)

    prepare = subparsers.add_parser('prepare', help=_("prepare dataset or model"))
    prepare_subparsers = prepare.add_subparsers(title='type', help=_("prepare type"))

    dataset = prepare_subparsers.add_parser('dataset', help=_("prepare dataset"))
    dataset.add_argument('type', type=DatasetType, help=_("dataset"))
    dataset.set_defaults(func=prepare_dataset)

    model = prepare_subparsers.add_parser('model', help=_("prepare model"))
    model.add_argument('type', choices=ModelType, type=ModelType, help=_("model"))
    model.add_argument('-e', '--nb_epoch', type=int, default=10, help=_("epoch count"))
    model.add_argument('--triplets', dest='triplets', action='store_true')
    model.set_defaults(triplets=False)
    model.add_argument('-d', '--dataset', choices=DatasetType, type=DatasetType, help=_("dataset"), default=DatasetType.market1501)
    model.set_defaults(func=prepare_model)

    args = parser.parse_args()

    if 'func' not in args:
        return parser.print_help()
    else:
        return args.func(args)
