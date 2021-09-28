import os
import time
from datetime import datetime
from functools import partial

import numpy as np
from tqdm import tqdm, trange

import jax
from jax import random, jit, value_and_grad
from jax import numpy as jnp
from jax.experimental import optimizers

from neural_tangents import stax

from ..classification.data import get_train_dataset
from ..utils import DataLoader, Logger, get_context_summary


def add_subparser(subparsers):
    parser = subparsers.add_parser("train", aliases=["tr"])
    parser.set_defaults(func=main)

    parser.add_argument("-m",   "--method",           choices=["gp", "tp"], required=True)
    parser.add_argument("-n",   "--network",          choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("-dr",  "--data-root",        type=str, default="./data")
    parser.add_argument("-dn",  "--data-name",        required=True)
    parser.add_argument("-cr",  "--ckpt-root",        type=str, default="./_ckpt/ens")
    parser.add_argument("-cn",  "--ckpt-name",        type=str, default=None)
    parser.add_argument("-ci",  "--ckpt-interval",    type=int, default=20)

    parser.add_argument("-nd",  "--num-data",         type=int, default=None)
    parser.add_argument("-nb",  "--num-batch",        type=int, default=500)

    parser.add_argument("-a",   "--alpha",            type=float, default=2.)
    parser.add_argument("-b",   "--beta",             type=float, default=2.)

    parser.add_argument("-nh",  "--num-hiddens",      type=int, default=4)
    parser.add_argument("-nc",  "--num-channels",     type=int, default=32)
    parser.add_argument("-act", "--activation",       choices=["erf", "relu"], default="relu")
    parser.add_argument("-ws",  "--w-std",            type=float, default=1.)
    parser.add_argument("-bs",  "--b-std",            type=float, default=0.)

    parser.add_argument("-opt", "--optimizer",        choices=["adam", "sgd"], default="adam")
    parser.add_argument("-lr",  "--lr",               type=float, default=1e-2)
    parser.add_argument("-e",   "--max-epoch",        type=int, default=300)

    parser.add_argument("-s",   "--seed",             type=int, default=10)
    parser.add_argument("-q",   "--quite",            default=False, action="store_true")
    parser.add_argument("-c",   "--comment",          type=str, default="")


@partial(jit, static_argnums=(1, 2))
def invgamma(key, alpha, beta):
    sigma = jnp.sqrt(beta / random.gamma(key, a=alpha)).item()
    return sigma


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


def get_train_step(apply_fn, get_params, opt_update):
    @value_and_grad
    def loss(params, x_batch, y_batch):
        logits = apply_fn(params, x_batch)
        return cross_entropy(logits, y_batch)

    @jit
    def train_step(i, opt_state, x_batch, y_batch):
        params = get_params(opt_state)
        v, g = loss(params, x_batch, y_batch)
        opt_state = opt_update(i, g, opt_state)
        return v, opt_state

    return train_step


def train_epoch(epoch, opt_state, train_loader, train_step, train_log):
    nll_list = []
    steps = len(train_loader) * (epoch - 1)
    log_interval = len(train_loader) // 4

    for idx, (x_batch, y_batch) in tqdm(enumerate(train_loader), desc="Train", leave=False, ncols=0):
        nll, opt_state = train_step(steps + idx, opt_state, x_batch, y_batch)
        nll_list.append(nll * x_batch.shape[0])
        if (idx + 1) % log_interval == 0:
            train_log(idx, nll)

    train_nll = (np.sum(np.array(nll_list)) / train_loader.num_data).item()
    return train_nll, opt_state


def main(args):
    # Dataset
    dataset = get_train_dataset(
        name=args.data_name, root=args.data_root,
        num_data=args.num_data, valid_prop=0.,
        normalize=True, onehot=True, seed=args.seed,
    )

    (x_train, y_train), (num_class, data_name, data_msg) = dataset
    num_train = x_train.shape[0]

    # Log and checkpoint
    if not args.ckpt_name:
        args.ckpt_name = f"{data_name}"
        args.ckpt_name += f"/{args.method}-{args.network}"
        args.ckpt_name += f"/nh{args.num_hiddens}-nc{args.num_channels}"
        if args.method == "svtp":
            args.ckpt_name += f"-a{args.alpha:.1f}-b{args.beta:.1f}"
        if args.comment:
            args.ckpt_name += f"/{args.comment}"
        else:
            args.ckpt_name += f"/{str(datetime.now().strftime('%y%m%d%H%M'))}"

    ckpt_dir = os.path.join(os.path.expanduser(args.ckpt_root), args.ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(ckpt_dir, quite=args.quite)

    try:
        key = random.PRNGKey(args.seed)

        # Model
        if args.method == "gp":
            last_w_std = 1.
        elif args.method == "tp":
            last_w_std = invgamma(key, args.alpha, args.beta)

        net_kwargs = dict(
            num_hiddens=args.num_hiddens, num_channels=args.num_channels, num_class=num_class,
            w_std=args.w_std, b_std=args.b_std, last_w_std=last_w_std,
        )
        init_fn, apply_fn = get_cnn(**net_kwargs)

        # Log
        np.save(os.path.join(ckpt_dir, "meta.npy"), vars(args))
        logger.log(get_context_summary(args, dict(
            num_class=num_class, num_train=num_train,
            data_name=data_name, data_msg=data_msg,
            last_w_std=last_w_std,
        )))

        # Optimizer
        if args.optimizer == "adam":
            opt_init, opt_update, get_params = optimizers.adam(args.lr)
        elif args.optimizer == "sgd":
            opt_init, opt_update, get_params = optimizers.sgd(args.lr)
        else:
            raise ValueError(f"Unsupported optimizer '{args.optimizer}'")

        params = init_fn(key, (-1, *x_train.shape[1:]))[1]
        opt_state = opt_init(params)

        def train_log(i, nelbo, log=True):
            print_str = f"nELBO: {nelbo:.5f}"
            if log:
                logger.log(f"       [{i:4d}]  {print_str}", is_tqdm=True)
            return print_str

        # Train
        train_step = get_train_step(apply_fn, get_params, opt_update)
        train_loader = DataLoader(x_train, y_train, batch_size=args.num_batch, shuffle=True)

        for epoch in trange(1, args.max_epoch + 1, desc="Epoch", ncols=0):
            train_nll, opt_state = train_epoch(epoch, opt_state, train_loader, train_step, train_log)
            logger.log(f"[{epoch:3d}]  NLL: {train_nll:.5f}", is_tqdm=True)

            if epoch % args.ckpt_interval == 0:
                params = get_params(opt_state)
                jnp.savez(os.path.join(ckpt_dir, f"{epoch:03d}.npz"), params=params, **net_kwargs)
                logger.log(f"[{epoch:3d}]  Checkpoint saved", is_tqdm=True)

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    except:
        import traceback
        logger.log(f"\n{traceback.format_exc()}\nStopped")

    finally:
        logger.close()
