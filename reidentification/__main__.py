#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reidentification experiments.

Main module - parse cli and run a functions.
"""

import argparse
from enum import Enum

from .i18n import _
from .evaluate import evaluate as run_evaluate
from .datasets import prepare_market1501
from .models import prepare_simple

class PrepareType(Enum):
    """Prepare type"""
    # datasets
    market1501 = 'market1501'

    # models
    simple = 'simple'

def run_prepare(args):
    """Prepare dataset or model."""
    print(_("Preparing {type}").format(type=args.type))
    if args.type == PrepareType.market1501:
        prepare_market1501()
    elif args.type == PrepareType.simple:
        prepare_simple(nb_epoch=args.nb_epoch)
    else:
        raise NotImplementedError()

def main():
    """Main function - parse command-line arguments and run function."""
    parser = argparse.ArgumentParser(description=_("Run reidentification experiments."))
    parser.add_argument('-e', '--nb_epoch', type=int, default=10, help=_("epoch count"))

    subparsers = parser.add_subparsers(title='actions', help=_("system actions"))

    evaluate = subparsers.add_parser('evaluate', help=_("evaluate model"))
    evaluate.set_defaults(func=run_evaluate)

    prepare = subparsers.add_parser('prepare', help=_("prepare dataset or model"))
    prepare.add_argument('type', choices=PrepareType, type=PrepareType, help=_("prepare type"))
    prepare.set_defaults(func=run_prepare)

    args = parser.parse_args()

    if 'func' not in args:
        return parser.print_help()
    else:
        return args.func(args)
