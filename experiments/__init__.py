from . import (
    classification,
)


def add_subparser(subparsers):
    classification.add_subparser(subparsers)
