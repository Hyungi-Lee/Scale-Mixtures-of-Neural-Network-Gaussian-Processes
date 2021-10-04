import os
import glob

from tqdm import tqdm

import numpy as np

import jax
from jax import random
from jax import numpy as jnp

import objax

from spax.models import SVSP
from spax.kernels import NNGPKernel
from spax.priors import GaussianPrior, InverseGammaPrior

from .data import get_test_dataset
from ..nt_kernels import get_cnn_kernel, get_conv_resnet_kernel
from ..utils import DataLoader, Checkpointer, Logger


def add_subparser(subparsers):
    parser = subparsers.add_parser("test", aliases=["ts"])
    parser.set_defaults(func=main)

    parser.add_argument("-dr", "--data-root",  type=str, default="./data")
    parser.add_argument("-dn", "--data-name",  required=True)
    parser.add_argument("-cd", "--ckpt-dir",   type=str, required=True)
    parser.add_argument("-ci", "--ckpt-index", type=int, default=None)
    parser.add_argument("-nd", "--num-data",   type=int, default=None)
    parser.add_argument("-nb", "--num-batch",  type=int, default=100)
    parser.add_argument("-ns", "--num-sample", type=int, default=10000)
    parser.add_argument("-s",  "--seed",       type=int, default=10)
    parser.add_argument("-q",  "--quite",      default=False, action="store_true")


def build_test_step(model, num_samples, jit=True):
    def test_step(key, x_batch, y_batch):
        nll, correct_count = model.test_acc_nll(key, x_batch, y_batch, num_samples)
        return nll, correct_count
    return objax.Jit(test_step, model.vars()) if jit else test_step


def test_epoch(key, test_loader, test_step):
    nll_list = []
    total_corrects = 0

    for x_batch, y_batch in tqdm(test_loader, desc="Test", leave=False, ncols=0):
        key, split_key = random.split(key)
        nll, corrects = test_step(split_key, x_batch, y_batch)
        nll_list.append(nll * x_batch.shape[0])
        total_corrects += corrects

    valid_nll = (np.sum(np.array(nll_list)) / test_loader.num_data).item()
    valid_acc = (total_corrects * 100 / test_loader.num_data).item()
    return valid_nll, valid_acc


def get_from_vars(saved_vars, key):
    names = saved_vars["names"]
    for i, name in enumerate(names):
        if key == name.split(".")[-1]:
            return saved_vars[str(i)]
    return None


def main(args):
    if args.ckpt_index is None:
        last_ckpt = sorted(glob.glob(os.path.join(args.ckpt_dir, Checkpointer.FILE_MATCH)))[-1]
        args.ckpt_index = int("".join(last_ckpt.split("/")[-1].split(".")[:-1]))

    # Dataset
    (x_test, y_test), (num_class, data_name) = get_test_dataset(
        name=args.data_name, root=args.data_root,
        num_data=args.num_data, normalize=True,
    )

    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)

    # Load
    saved_vars = jnp.load(os.path.join(args.ckpt_dir, Checkpointer.FILE_FORMAT.format(args.ckpt_index)))
    context = jnp.load(os.path.join(args.ckpt_dir, "meta.npy"), allow_pickle=True).item()

    a = get_from_vars(saved_vars, "a")
    b = get_from_vars(saved_vars, "b")
    w_std = get_from_vars(saved_vars, "w_std")
    b_std = get_from_vars(saved_vars, "b_std")
    last_w_std = get_from_vars(saved_vars, "last_w_std")
    inducing_points = get_from_vars(saved_vars, "inducing_variable")
    q_mu = get_from_vars(saved_vars, "q_mu")
    q_sqrt = get_from_vars(saved_vars, "q_sqrt")

    # Context
    method = context["method"]
    network = context["network"]
    num_hiddens = context["num_hiddens"]
    activation = context["activation"]
    if "alpha" in context:
        alpha = context["alpha"]
        beta = context["beta"]

    if last_w_std is None:
        last_w_std = np.array(context["last_w_std"])

    # Log
    log_dir = os.path.join(args.ckpt_dir, "test")
    log_name = f"{method}-{network}-{data_name.replace('/', '-')}-{args.ckpt_index}.log"
    logger = Logger(log_dir, log_name, quite=args.quite)

    # Resize
    h, w, c = inducing_points.shape[1:]
    x_test = jax.image.resize(x_test, (x_test.shape[0], h, w, c), method="bilinear")

    # Kernel
    if network == "cnn":
        base_kernel_fn = get_cnn_kernel
    else:
        network = "resnet"
        base_kernel_fn = get_conv_resnet_kernel

    def get_kernel_fn(w_std, b_std, last_w_std):
        return base_kernel_fn(
            num_hiddens, num_class, activation,
            w_std=w_std, b_std=b_std, last_w_std=last_w_std,
        )

    kernel = NNGPKernel(get_kernel_fn, 0, 0, 0)

    if method == "svgp":
        prior = GaussianPrior()
    elif method == "svtp":
        prior = InverseGammaPrior(alpha, beta)

    model = SVSP(prior, kernel, inducing_points, num_latent_gps=num_class)
    model.kernel.w_std.assign(jnp.array(w_std))
    model.kernel.b_std.assign(jnp.array(b_std))
    model.kernel.last_w_std.assign(jnp.array(last_w_std))
    model.q_mu.assign(jnp.array(q_mu))
    model.q_sqrt.assign(jnp.array(q_sqrt))
    if method == "svtp":
        model.prior.a.assign(jnp.array(a))
        model.prior.b.assign(jnp.array(b))

    logger.log(f"Data: {data_name}")
    logger.log(f"Epoch: {args.ckpt_index}")
    logger.log("\n" + str(model.vars()) + "\n")

    # Build functions
    test_step = build_test_step(model, args.num_sample)
    test_loader = DataLoader(x_test, y_test, batch_size=args.num_batch, shuffle=False)

    # Test
    key = random.PRNGKey(args.seed)

    test_nll, test_acc = test_epoch(key, test_loader, test_step)

    logger.log(f"NLL: {test_nll:.5f}  ACC: {test_acc:.2f}\n")
    logger.close()
