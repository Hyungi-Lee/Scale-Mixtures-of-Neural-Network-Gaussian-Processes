import os
import glob
from datetime import datetime

from tqdm import tqdm

from jax import random
from jax import numpy as jnp

import objax
from objax.optimizer import SGD, Adam

from spax.models import SVSP
from spax.kernels import NNGPKernel
from spax.priors import GaussianPrior, InverseGammaPrior

from .data import get_test_dataset, datasets
from ..nt_kernels import get_mlp_kernel, get_cnn_kernel, get_resnet_kernel
from ..utils import TestBatch, Checkpointer, Logger


def add_subparser(subparsers):
    parser = subparsers.add_parser("test", aliases=["ts"])
    parser.set_defaults(func=main)

    # parser.add_argument("-m",   "--method",      choices=["svgp", "svtp"], required=None)
    # parser.add_argument("-n",   "--network",     choices=["cnn", "resnet", "mlp"], default=None)

    parser.add_argument("-dn",  "--data-name",   choices=datasets, required=True)
    parser.add_argument("-dr",  "--data-root",   type=str, default="./data")
    parser.add_argument("-cr",  "--ckpt-root",   type=str, default="./ckpt")
    parser.add_argument("-cn",  "--ckpt-name",   type=str, required=True)
    parser.add_argument("-ci",  "--ckpt-index",  type=int, default=None)
    parser.add_argument("-er",  "--eval-root",   type=str, default="./eval")
    parser.add_argument("-en",  "--eval-name",   type=str, default=None)

    parser.add_argument("-nts", "--num-test",    type=int, default=None)
    parser.add_argument("-ns",  "--num-sample",  type=int, default=10000)

    # parser.add_argument("-nh",  "--num-hiddens", type=int, default=4)
    # parser.add_argument("-act", "--activation",  choices=["erf", "relu"], default="relu")

    parser.add_argument("-q",   "--quite",       default=False, action="store_true")


def build_test_step(model, num_batch, num_samples, jit=True):
    def test_step(key, x_batch, y_batch):
        nll, correct_count = model.test_acc_nll(key, x_batch, y_batch, num_batch, num_samples)
        return nll, correct_count
    return objax.Jit(test_step, model.vars()) if jit else test_step


def test(key, test_batches, test_step, num_test):
    test_nll_list = []
    total_corrects = 0

    for x_batch, y_batch in tqdm(test_batches, desc="Test", leave=False, ncols=0):
        key, split_key = random.split(key)
        nll, corrects = test_step(split_key, x_batch, y_batch)
        test_nll_list.append(nll)
        total_corrects += corrects

    test_nll = (jnp.sum(jnp.array(test_nll_list)) / num_test).item()
    test_acc = (total_corrects / num_test).item()
    return test_nll, test_acc


def get_from_vars(saved_vars, key):
    names = saved_vars["names"]
    for i, name in enumerate(names):
        if key == name.split(".")[-1]:
            return saved_vars[str(i)]
    return None


def main(args):
    # Log

    # Temp
    ckpt_dir = os.path.join(os.path.expanduser(args.ckpt_root), args.ckpt_name)
    # eval_dir = os.path.join(os.path.expanduser(args.eval_root), args.eval_name)
    # logger = Logger(eval_dir, quite=args.quite)
    if args.ckpt_index is None:
        args.ckpt_indx = int("".join(sorted(glob.glob(os.path.join(ckpt_dir, Checkpointer.FILE_MATCH)))[-1].split(".")[:-1]))

    saved_vars = jnp.load(os.path.join(ckpt_dir, Checkpointer.FILE_FORMAT.format(args.ckpt_index)))
    context_info = jnp.load(os.path.join(ckpt_dir, "meta.npy"), allow_pickle=True).item()

    # Model params
    a = get_from_vars(saved_vars, "a")
    b = get_from_vars(saved_vars, "b")
    w_std = get_from_vars(saved_vars, "w_std")
    b_std = get_from_vars(saved_vars, "b_std")
    last_w_std = get_from_vars(saved_vars, "last_w_std")
    inducing_points = get_from_vars(saved_vars, "inducing_variable")
    q_mu = get_from_vars(saved_vars, "q_mu")
    q_sqrt = get_from_vars(saved_vars, "q_sqrt")

    # Context
    dataset_stat = context_info["dataset"]["stat"]
    args.method = context_info["args"]["method"]
    args.network = context_info["args"]["network"]
    args.num_hiddens = context_info["args"]["num_hiddens"]
    args.activation = context_info["args"]["activation"]
    if "alpha" in context_info["args"]:
        args.alpha = context_info["args"]["alpha"]
        args.beta = context_info["args"]["beta"]

    class TempLogger:
        def log(self, *args, **kwargs):
            print(*args)
        def close(self):
            pass

    logger = TempLogger()

    # Dataset
    x_test, y_test, dataset_info = get_test_dataset(
        name=args.data_name, root=args.data_root,
        num_test=args.num_test, normalize=True,
        dataset_stat=dataset_stat, one_hot=True,
    )

    num_class = dataset_info["num_class"]
    num_test = x_test.shape[0]

    # Kernel
    if dataset_info["type"] == "image":
        if args.network == "cnn":
            args.network = "cnn"
            base_kernel_fn = get_cnn_kernel
        else:
            args.network = "resnet"
            base_kernel_fn = get_resnet_kernel
    elif dataset_info["type"] == "feature":
        args.network = "mlp"
        base_kernel_fn = get_mlp_kernel

    def get_kernel_fn(w_std, b_std, last_w_std):
        return base_kernel_fn(
            args.num_hiddens, num_class, args.activation,
            w_std=w_std, b_std=b_std, last_w_std=last_w_std,
        )

    kernel = NNGPKernel(get_kernel_fn, 0, 0, 0)

    # Model

    if args.method == "svgp":
        prior = GaussianPrior()
    elif args.method == "svtp":
        prior = InverseGammaPrior(args.alpha, args.beta)

    model = SVSP(prior, kernel, inducing_points, num_latent_gps=num_class)
    model.kernel.w_std.assign(w_std)
    model.kernel.b_std.assign(b_std)
    model.kernel.last_w_std.assign(last_w_std)
    model.q_mu.assign(q_mu)
    model.q_sqrt.assign(q_sqrt)
    if args.method == "svtp":
        model.prior.a.assign(a)
        model.prior.b.assign(b)

    # print(model.vars(), end="\n\n")

    # Build functions
    num_test_batch = 100

    test_step = build_test_step(model, num_test_batch, args.num_sample)
    test_batches = TestBatch(x_test, y_test, num_test_batch)

    # Train
    key = random.PRNGKey(10)

    test_nll, test_acc = test(key, test_batches, test_step, num_test)
    logger.log(f"NLL: {test_nll:.5f}  ACC: {test_acc:.4f}\n")
    logger.close()
