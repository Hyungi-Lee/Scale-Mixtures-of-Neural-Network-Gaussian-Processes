import os
import argparse
import warnings

# Filter warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Only for environment preparation
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-g", "--gpu", type=int, nargs="*")
parser.add_argument("-f", "--fraction", type=float)
args, main_args = parser.parse_known_args()

# Set up environment
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in args.gpu)

if args.fraction is not None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.fraction)


# Main
import experiments


def main(raw_args):
    parser = argparse.ArgumentParser(description="Scale Mixtures of NNGP")
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    experiments.add_subparser(subparsers)

    args = parser.parse_args(raw_args)
    main_func = args.func

    try:
        main_func(args)
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main(main_args)
