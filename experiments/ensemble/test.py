import os
import glob

from tqdm import tqdm

import numpy as np

import jax
from jax import jit
from jax import numpy as jnp

from neural_tangents import stax

from ..classification.data import get_test_dataset
from ..utils import DataLoader, Logger


def add_subparser(subparsers):
    parser = subparsers.add_parser("test", aliases=["ts"])
    parser.set_defaults(func=main)

    parser.add_argument("-dr", "--data-root",  type=str, default="./data")
    parser.add_argument("-dn", "--data-name",  required=True)
    # parser.add_argument("-cd", "--ckpt-dir",   type=str, required=True)
    parser.add_argument("-cd", "--ckpt-dir",   type=str, required=True, nargs="+")
    parser.add_argument("-nd", "--num-data",   type=int, default=None)
    parser.add_argument("-nb", "--num-batch",  type=int, default=100)
    # parser.add_argument("-s",  "--seed",       type=int, default=10)
    parser.add_argument("-q",  "--quite",      default=False, action="store_true")


def get_cnn(num_hiddens, num_channels, num_class, w_std=1., b_std=0., last_w_std=1.):
    layers = []
    for _ in range(num_hiddens):
        layers.append(stax.Conv(num_channels, (3, 3), (1, 1), "SAME", W_std=w_std, b_std=b_std))
        layers.append(stax.Relu())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(num_class, W_std=last_w_std))
    init_fn, apply_fn, _ = stax.serial(*layers)
    return init_fn, jit(apply_fn)


@jit
def cross_entropy(logits, y):
    return -jnp.mean(jax.nn.log_softmax(logits) * y)


def test_epoch(test_loader, apply_fns, paramss):
    nll = 0.
    corrects = 0

    for x_batch, y_batch in tqdm(test_loader, desc="Test", leave=False, ncols=0):
        logits = [apply_fn(params, x_batch)[None, :] for apply_fn, params in zip(apply_fns, paramss)]
        logits = np.mean(np.vstack(logits), axis=0)

        nll += cross_entropy(logits, y_batch) * x_batch.shape[0]
        corrects += np.sum((np.argmax(logits, axis=1) == np.argmax(y_batch, axis=1)))

    test_nll = (nll / test_loader.num_data).item()
    test_acc = (corrects * 100 / test_loader.num_data).item()
    return test_nll, test_acc


def main(args):

    # Dataset
    (x_test, y_test), (num_class, data_name) = get_test_dataset(
        name=args.data_name, root=args.data_root,
        num_data=args.num_data, onehot=True, normalize=True,
    )

    # Load
    apply_fns = []
    paramss = []
    for ckpt_dir in args.ckpt_dir:
        last_ckpt = sorted(glob.glob(os.path.join(ckpt_dir, "*.npy")))[-2]
        ckpt_index = int("".join(last_ckpt.split("/")[-1].split(".")[:-1]))
        ckpt = jnp.load(os.path.join(ckpt_dir, "{:03d}.npy".format(ckpt_index)), allow_pickle=True)
        params, net_args = ckpt
        apply_fn = get_cnn(*net_args)[1]
        apply_fns.append(apply_fn)
        paramss.append(params)

    test_loader = DataLoader(x_test, y_test, batch_size=args.num_batch, shuffle=False)
    test_nll, test_acc = test_epoch(test_loader, apply_fns, paramss)
    print(f"{test_nll = :.6f}   {test_acc = :.2f}")
