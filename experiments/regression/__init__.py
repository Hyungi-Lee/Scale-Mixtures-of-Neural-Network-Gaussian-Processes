from . import (
    test,
    train,
    data,
)


def add_subparser(subparsers):
    parser = subparsers.add_parser("regression", aliases=["reg"])
    subparsers = parser.add_subparsers(metavar="ops")

    train.add_subparser(subparsers)
    # train1.add_subparser(subparsers)
    # test.add_subparser(subparsers)
    test.add_subparser(subparsers)
