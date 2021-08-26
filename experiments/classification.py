def add_subparser(subparsers):
    parser = subparsers.add_parser("classification", aliases=["cls"], help="classification")
    parser.set_defaults(func=main)

    parser.add_argument("dataset", choices=["t"])


def main(dataset):
    print(dataset)
