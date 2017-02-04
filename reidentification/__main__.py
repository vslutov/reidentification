#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reidentification experiments.

Main module - parse cli and run a functions.
"""

import argparse

from .i18n import _
from .evaluate import evaluate as run_evaluate

def main():
    """Main function - parse command-line arguments and run function."""
    parser = argparse.ArgumentParser(description=_("Run reidentification experiments."))
    subparsers = parser.add_subparsers(title='actions', help=_("system actions"))

    evaluate = subparsers.add_parser('evaluate', help=_("run evaluate"))
    evaluate.set_defaults(func=run_evaluate)

    args = parser.parse_args()

    if 'func' not in args:
        return parser.print_help()
    else:
        return args.func(args)
