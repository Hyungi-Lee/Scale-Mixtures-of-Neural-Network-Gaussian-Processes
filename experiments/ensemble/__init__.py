from . import (
    train,
    test,
)


def add_subparser(subparsers):
    parser = subparsers.add_parser("ensemble", aliases=["ens"])
    subparsers = parser.add_subparsers(metavar="ops")

    train.add_subparser(subparsers)
    test.add_subparser(subparsers)
