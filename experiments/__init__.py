from . import (
    classification,
    regression,
    ensemble,
)


def add_subparser(subparsers):
    classification.add_subparser(subparsers)
    regression.add_subparser(subparsers)
    ensemble.add_subparser(subparsers)
