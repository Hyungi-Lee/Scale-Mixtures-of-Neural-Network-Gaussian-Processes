import os
import glob

import numpy as np

import objax
from jax import numpy as jnp

from spax.models import SPR
from spax.kernels import NNGPKernel
from spax.likelihoods import GaussianLikelihood, StudentTLikelihood

from .data import get_dataset, permute_dataset, split_dataset
from ..nt_kernels import get_mlp_kernel, get_dense_resnet_kernel
from ..utils import Checkpointer, Logger


def add_subparser(subparsers):
    parser = subparsers.add_parser("test", aliases=["ts"])
    parser.set_defaults(func=main)

    parser.add_argument("-dr",  "--data-root",        type=str, default="./data")
    parser.add_argument("-cd",  "--ckpt-dir",         type=str, required=True)
    parser.add_argument("-ci",  "--ckpt-index",       type=int, default=None)
    parser.add_argument("-vp",  "--valid-prop",       type=float, default=0.1)
    parser.add_argument("-nd",  "--num-data",         type=int, default=None)
    parser.add_argument("-s",   "--seed",             type=int, default=10)
    parser.add_argument("-q",   "--quite",            default=False, action="store_true")


def build_test_step(model, x_test, y_test, jit=True):
    def test_step():
        nll = model.test_nll(x_test, y_test)
        return nll
    return objax.Jit(test_step, model.vars()) if jit else test_step


def get_from_vars(saved_vars, key):
    names = saved_vars["names"]
    for i, name in enumerate(names):
        if key == name.split(".")[-1]:
            return saved_vars[str(i)]
    return None


def main(args):
    if args.ckpt_index is None:
        ckpts = glob.glob(os.path.join(args.ckpt_dir, Checkpointer.FILE_MATCH))
        args.ckpt_index = sorted([int("".join(ckpt.split("/")[-1].split(".")[:-1])) for ckpt in ckpts])[-1]

    # Log
    saved_vars = jnp.load(os.path.join(args.ckpt_dir, Checkpointer.FILE_FORMAT.format(args.ckpt_index)))
    context = jnp.load(os.path.join(args.ckpt_dir, "meta.npy"), allow_pickle=True).item()["args"]

    log_dir = os.path.join(args.ckpt_dir, "test")
    log_name = f"test.log"
    logger = Logger(log_dir, log_name, quite=args.quite)

    try:
        # Context
        method = context["method"]
        network = context["network"]
        num_hiddens = context["num_hiddens"]
        activation = context["activation"]
        data_name = context["data_name"]

        # Dataset
        x, y = get_dataset(name=data_name, root=args.data_root)
        x, y = permute_dataset(x, y, seed=10)
        splits = split_dataset(x, y, train=0.8, valid=0.1, test=0.1)
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (y_std, y_mean) = splits

        num_train = x_train.shape[0]
        num_valid = x_valid.shape[0]
        num_test = x_test.shape[0]

        x_train_valid = np.concatenate([x_train, x_valid], axis=0)
        y_train_valid = np.concatenate([y_train, y_valid], axis=0)

        x_train_valid, y_train_valid = permute_dataset(x_train_valid, y_train_valid, seed=args.seed)
        x_train, x_valid = x_train_valid[:num_train], x_train_valid[num_train:]
        y_train, y_valid = y_train_valid[:num_train], y_train_valid[num_train:]

        x_train, y_train = jnp.array(x_train), jnp.array(y_train)
        x_valid, y_valid = jnp.array(x_valid), jnp.array(y_valid)
        x_test, y_test = jnp.array(x_test), jnp.array(y_test)
        y_std, y_mean = jnp.array(y_std), jnp.array(y_mean)

        a = get_from_vars(saved_vars, "a")
        b = get_from_vars(saved_vars, "b")
        w_std = get_from_vars(saved_vars, "w_std")
        b_std = get_from_vars(saved_vars, "b_std")
        last_w_std = get_from_vars(saved_vars, "last_w_std")
        eps = get_from_vars(saved_vars, "eps")
        if eps is None:
            eps = get_from_vars(saved_vars, "diag_reg")

        if last_w_std is None:
            last_w_std = np.array(context["last_w_std"])

        # Model
        if network is None or network == "mlp":
            network = "mlp"
            base_kernel_fn = get_mlp_kernel
        elif network == "resnet":
            network = "resnet"
            base_kernel_fn = get_dense_resnet_kernel
        else:
            raise ValueError(f"Unsupported network '{network}'")

        def get_kernel_fn(w_std, b_std, last_w_std):
            kernel_fn = base_kernel_fn(
                num_hiddens, act=activation,
                w_std=w_std, b_std=b_std, last_w_std=last_w_std,
            )
            return kernel_fn

        kernel = NNGPKernel(get_kernel_fn, w_std, b_std, last_w_std)
        if method == "gp":
            likelihood = GaussianLikelihood()
        elif method == "tp":
            likelihood = StudentTLikelihood(1, 1)

        model = SPR(kernel, likelihood, x_train, y_train, y_mean, y_std, eps=1)

        model.eps.assign(jnp.array(eps))
        model.kernel.w_std.assign(jnp.array(w_std))
        model.kernel.b_std.assign(jnp.array(b_std))
        model.kernel.last_w_std.assign(jnp.array(last_w_std))
        model.likelihood.a.assign(jnp.array(a))
        model.likelihood.b.assign(jnp.array(b))

        # Build functions
        test_step = build_test_step(model, x_test, y_test)
        test_nll = test_step()
        logger.log(f"NLL: {test_nll:.5f}")

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    except:
        import traceback
        logger.log(f"\n{traceback.format_exc()}\nStopped")

    finally:
        logger.close()
