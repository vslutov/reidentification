#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reidentification experiments.

Main module - parse cli and run a functions.
"""

import argparse

from .i18n import _
from .evaluate import evaluate as run_evaluate
from .datasets import prepare_market1501

def run_prepare(args):
    """Prepare dataset or model."""
    print(_("Preparing {part}").format(part=args.part))
    if args.part == 'market1501':
        prepare_market1501()
    else:
        raise NotImplementedError()

def main():
    """Main function - parse command-line arguments and run function."""
    parser = argparse.ArgumentParser(description=_("Run reidentification experiments."))
    subparsers = parser.add_subparsers(title='actions', help=_("system actions"))

    evaluate = subparsers.add_parser('evaluate', help=_("evaluate model"))
    evaluate.set_defaults(func=run_evaluate)

    prepare = subparsers.add_parser('prepare', help=_("prepare dataset or model"))
    prepare.add_argument('part', choices=['market1501'], help=_("what part should prepare"))
    prepare.set_defaults(func=run_prepare)

    args = parser.parse_args()

    if 'func' not in args:
        return parser.print_help()
    else:
        return args.func(args)
