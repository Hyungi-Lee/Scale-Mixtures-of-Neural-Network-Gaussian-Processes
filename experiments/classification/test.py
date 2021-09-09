import os
from datetime import datetime

from tqdm import tqdm

from jax import random
from jax import numpy as jnp

import objax
from objax.optimizer import SGD, Adam

from neural_tangents import stax

from spax.models import SVSP
from spax.kernels import NNGPKernel
from spax.priors import GaussianPrior, InverseGammaPrior

from .data import get_test_dataset, datasets
from ..utils import TrainBatch, TestBatch, Checkpointer, Logger


def add_subparser(subparsers):
    parser = subparsers.add_parser("test", aliases=["ts"])
    parser.set_defaults(func=main)

    parser.add_argument("-m",    "--method",           choices=["svgp", "svtp"], required=True)
    parser.add_argument("-n",    "--network",          choices=["cnn", "resnet", "mlp"], default=None)
    parser.add_argument("-dn",   "--data-name",        choices=datasets, required=True)
    parser.add_argument("-tdn",  "--test-data-name",   choices=datasets, default=None)
    parser.add_argument("-dr",   "--data-root",        type=str, default="./data")
    parser.add_argument("-cr",   "--ckpt-root",        type=str, default="./ckpt")
    parser.add_argument("-cn",   "--ckpt-name",        type=str, required=True)

    parser.add_argument("-nts",  "--num-test",         type=int, default=None)
    parser.add_argument("-ntss", "--num-test-sample",  type=int, default=10000)

    parser.add_argument("-nh",   "--num-hiddens",      type=int, default=4)
    parser.add_argument("-act",  "--activation",       choices=["erf", "relu"], default="relu")

    parser.add_argument("-s",    "--seed",             type=int, default=10)
    parser.add_argument("-q",    "--quite",            default=False, action="store_true")


def get_act_class(act):
    if act == "relu":
        return stax.Relu
    elif act == "erf":
        return stax.Erf
    else:
        raise KeyError("Unsupported act '{}'".format(act))


def get_mlp_kernel(num_hiddens, num_classes, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    layers = []
    for _ in range(num_hiddens):
        layers.append(stax.Dense(512, W_std=w_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Dense(num_classes, W_std=last_w_std))

    _, _, kernel_fn = stax.serial(*layers)
    return  kernel_fn


def get_cnn_kernel(num_hiddens, num_classes, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    layers = []
    for _ in range(num_hiddens):
        layers.append(stax.Conv(1, (3, 3), (1, 1), "SAME", W_std=w_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(num_classes, W_std=last_w_std))

    _, _, kernel_fn = stax.serial(*layers)
    return  kernel_fn


def get_resnet_kernel(
    depth,
    class_num,
    act="relu",
    w_std=1.,
    b_std=0.,
    last_w_std=1.,
):
    act_class = get_act_class(act)

    def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
        Main = stax.serial(
            act_class(), stax.Conv(channels, (3, 3), strides, padding="SAME", W_std=w_std, b_std=b_std),
            act_class(), stax.Conv(channels, (3, 3), padding="SAME", W_std=w_std, b_std=b_std))
        Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
            channels, (3, 3), strides, padding="SAME", W_std=w_std, b_std=b_std)
        return stax.serial(stax.FanOut(2),
                           stax.parallel(Main, Shortcut),
                           stax.FanInSum())

    def WideResnetGroup(n, channels, strides=(1, 1)):
        blocks = []
        blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
        for _ in range(n - 1):
            blocks += [WideResnetBlock(channels, (1, 1))]
        return stax.serial(*blocks)

    def WideResnet(block_size, k, num_classes):
        return stax.serial(
            stax.Conv(16, (3, 3), padding="SAME", W_std=w_std, b_std=b_std),
            WideResnetGroup(block_size, int(8 * k)),
            WideResnetGroup(block_size, int(16 * k), (2, 2)),
            WideResnetGroup(block_size, int(32 * k), (2, 2)),
            WideResnetGroup(block_size, int(64 * k), (2, 2)),
            # stax.AvgPool((8, 8)),
            stax.Flatten(),
            stax.Dense(num_classes, W_std=last_w_std))

    _, _, kernel_fn = WideResnet(block_size=depth, k=1, num_classes=class_num)
    return kernel_fn


def build_test_batch(model, num_batch, num_samples, jit=True):
    def test_batch(key, x_batch, y_batch):
        nll, correct_count = model.test_acc_nll(key, x_batch, y_batch, num_batch, num_samples)
        return nll, correct_count
    return objax.Jit(test_batch, model.vars()) if jit else test_batch


def main(args):
    # Log
    if not args.ckpt_name:
        if args.test_data_name:
            args.ckpt_name = args.test_data_name
        else:
            args.ckpt_name = args.data_name
        args.ckpt_name += f"/{args.method}"
        args.ckpt_name += f"/ni{args.num_inducing}-nh{args.num_hiddens}-ws{args.w_std:.1f}-bs{args.b_std:.1f}-ls{args.last_w_std:.1f}"
        if args.method == "svtp":
            args.ckpt_name += f"-a{args.alpha:.1f}-b{args.beta:.1f}"
        if args.kmeans:
            args.ckpt_name += "-km"
        args.ckpt_name += f"-s{args.seed}"
        args.ckpt_name += f"/{str(datetime.now().strftime('%y%m%d%H%M'))}"

    ckpt_dir = os.path.join(os.path.expanduser(args.ckpt_root), args.ckpt_name)
    checkpointer = Checkpointer(ckpt_dir, keep_ckpts=3)
    logger = Logger(ckpt_dir, quite=args.quite)

    # Dataset
    dataset = get_dataset(
        name=args.data_name, root=args.data_root,
        num_train=args.num_train, num_test=args.num_test,
        normalize=True, one_hot=True, seed=args.seed,
    )

    if args.test_data_name is None:
        x_train, y_train, x_test, y_test, dataset_info = dataset
    else:
        x_train, y_train, _, _, dataset_info = dataset
        _, _, x_test, y_test, _ = get_dataset(
            name=args.test_data_name, root=args.data_root,
            num_train=args.num_train, num_test=args.num_test,
            normalize=True, dataset_stat=dataset_info["stat"],
            one_hot=True, seed=args.seed,
        )

    num_class = dataset_info["num_classes"]
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    # Kernel
    if dataset_info["type"] == "image":
        assert args.network is None or args.network in ["cnn", "resnet"], "Only CNN and ResNet are supported for image dataset"
        if args.network is None or args.network == "cnn":
            args.network = "cnn"
            base_kernel_fn = get_cnn_kernel
        else:
            args.network = "resnet"
            base_kernel_fn = get_resnet_kernel
    elif dataset_info["type"] == "feature":
        assert args.network is None or args.network in ["mlp"], "Only MLP is supported for feature dataset"
        args.network = "mlp"
        base_kernel_fn = get_mlp_kernel
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    def get_kernel_fn(w_std, b_std, last_w_std):
        return base_kernel_fn(
            args.num_hiddens, num_class, args.activation,
            w_std=w_std, b_std=b_std, last_w_std=last_w_std,
        )

    kernel = NNGPKernel(get_kernel_fn, args.w_std, args.b_std, args.last_w_std)

    # Model
    if args.kmeans:
        inducing_points = get_inducing_points(x_train, y_train, args.num_inducing, num_class)
    else:
        inducing_points = x_train[:args.num_inducing]

    if args.method == "svgp":
        prior = GaussianPrior()
    elif args.method == "svtp":
        prior = InverseGammaPrior(args.alpha, args.beta)
    else:
        raise ValueError(f"Unsupported method '{args.method}'")

    model = SVSP(prior, kernel, inducing_points, num_latent_gps=num_class)

    print()
    print(model.vars(), end="\n\n")

    # Optimizer
    if args.optimizer == "adam":
        optimizer = Adam(model.vars())
    elif args.optimizer == "sgd":
        optimizer = SGD(model.vars())
    else:
        raise ValueError(f"Unsupported optimizer '{args.optimizer}'")

    # Build functions
    num_test_batch = 100
    debug_mode = False

    train_batch = build_train_batch(model, optimizer, args.learning_rate, args.num_batch, num_train, args.num_train_sample, jit=not debug_mode)
    test_batch = build_test_batch(model, num_test_batch, args.num_test_sample, jit=not debug_mode)

    train_batches = TrainBatch(x_train, y_train, args.num_batch, args.steps, args.seed)
    test_batches = TestBatch(x_test, y_test, num_test_batch)

    # Log
    logger.log("Args:")
    logger.log("  method          : ", args.method)
    logger.log("  network         : ", args.network)
    logger.log("  data-name       : ", args.test_data_name)
    logger.log("  data-root       : ", args.data_root)
    logger.log("  ckpt-root       : ", args.ckpt_root)
    logger.log("  ckpt-name       : ", args.ckpt_name)
    logger.log("  num-test        : ", num_test)
    logger.log("  num-inducing    : ", args.num_inducing)
    logger.log("  num-sample      : ", args.num_test_sample)
    logger.log("  alpha           : ", args.alpha)
    logger.log("  beta            : ", args.beta)
    logger.log("  num-hiddens     : ", args.num_hiddens)
    logger.log("  activation      : ", args.activation)
    logger.log("  w-std           : ", args.w_std)
    logger.log("  b-std           : ", args.b_std)
    logger.log("  last-w-std      : ", args.last_w_std)
    logger.log("  optimizer       : ", args.optimizer)
    logger.log("  learning-rate   : ", args.learning_rate)
    logger.log("  steps           : ", args.steps)
    logger.log("  kmeans          : ", args.kmeans)
    logger.log("  seed            : ", args.seed)
    logger.log("  print-interval  : ", args.print_interval)
    logger.log("  test-interval   : ", args.test_interval)
    logger.log("")

    # Train
    key = random.PRNGKey(args.seed)

    best_model_acc = 0.
    best_model_nll = float("inf")
    best_step = 0
    best_print_str = ""

    for i, (x_batch, y_batch) in tqdm(enumerate(train_batches), total=args.steps, ncols=0):
        key, split_key = random.split(key)
        n_elbo = train_batch(split_key, x_batch, y_batch)

        if i % args.print_interval == 0:
            ws, bs, ls = model.kernel.get_params()

            if args.method == "svtp":
                ia, ib = (model.prior.a.safe_value, model.prior.b.safe_value)
                print_str = f"nELBO: {n_elbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}  a: {ia:.4f}  b: {ib:.4f}"
            else:
                print_str = f"nELBO: {n_elbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}"

            logger.log(f"[{i:5d}] " + print_str, is_tqdm=True)

        if i % args.test_interval == 0 or i == args.steps - 1:
            key, split_key = random.split(key)
            test_nll_list = []
            total_corrects = 0

            for x_batch, y_batch in tqdm(test_batches, leave=False, ncols=0):
                key, split_key = random.split(key)
                nll, corrects = test_batch(split_key, x_batch, y_batch)
                test_nll_list.append(nll)
                total_corrects += corrects

            test_nll = (jnp.sum(jnp.array(test_nll_list)) / num_test).item()
            test_acc = (total_corrects / num_test).item()

            logger.log(f"[{i:5d}] NLL: {test_nll:.5f}  ACC: {test_acc:.4f}", is_tqdm=True)

            if test_acc > best_model_acc or (jnp.allclose(test_acc, best_model_acc) and test_nll > best_model_nll):
                best_step, best_model_acc, best_model_nll = i, test_acc, test_nll
                best_print_str = print_str
                checkpointer.save(model.vars() + optimizer.vars(), i)
                logger.log(f"[{i:5d}] Updated: NLL: {test_nll:.5f}  ACC: {test_acc:.4f}", is_tqdm=True)

    logger.log("")
    logger.log(f"[{best_step:5d}] NLL: {best_model_nll:.5f}  ACC: {best_model_acc:.4f} " + best_print_str)
    logger.log("")

    logger.close()
