import os
from datetime import datetime

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

    parser.add_argument("-vp",  "--valid-prop",       type=float, default=0.1)
    parser.add_argument("-nd",  "--num-data",         type=int, default=None)
    parser.add_argument("-nb",  "--num-batch",        type=int, default=250)

    parser.add_argument("-a",   "--alpha",            type=float, default=2.)
    parser.add_argument("-b",   "--beta",             type=float, default=2.)

    parser.add_argument("-nh",  "--num-hiddens",      type=int, default=4)
    parser.add_argument("-nc",  "--num-channels",     type=int, default=32)
    parser.add_argument("-act", "--activation",       choices=["erf", "relu"], default="relu")
    parser.add_argument("-ws",  "--w-std",            type=float, default=1.)
    parser.add_argument("-bs",  "--b-std",            type=float, default=0.)
    parser.add_argument("-eps", "--epsilon",          type=float, default=1e-6)

    parser.add_argument("-opt", "--optimizer",        choices=["adam", "sgd"], default="adam")
    parser.add_argument("-lr",  "--lr",               type=float, default=1e-2)
    parser.add_argument("-e",   "--max-epoch",        type=int, default=100)

    parser.add_argument("-s",   "--seed",             type=int, default=10)
    parser.add_argument("-q",   "--quite",            default=False, action="store_true")
    parser.add_argument("-c",   "--comment",          type=str, default="")


def invgamma(key, alpha, beta):
    sigma = np.sqrt(beta / random.gamma(key, a=alpha)).item()
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


def build_train_step(apply_fn, get_params, opt_update):
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


def build_valid_step(apply_fn):
    @jit
    def valid_step(params, x_batch, y_batch):
        logits = apply_fn(params, x_batch)
        nll = cross_entropy(logits, y_batch)
        corrects = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y_batch, axis=-1))
        return nll, corrects
    return valid_step


def train_epoch(epoch, opt_state, train_loader, train_step, train_log):
    total_nll = 0.
    steps = len(train_loader) * (epoch - 1)
    log_interval = len(train_loader) // 4

    for idx, (x_batch, y_batch) in tqdm(enumerate(train_loader), desc="Train", leave=False, ncols=0):
        nll, opt_state = train_step(steps + idx, opt_state, x_batch, y_batch)
        total_nll += nll.item() * x_batch.shape[0]
        if (idx + 1) % log_interval == 0:
            train_log(idx + 1, nll)

    train_nll = total_nll / train_loader.num_data
    return train_nll, opt_state


def valid_epoch(params, valid_loader, valid_step):
    total_nll = 0.
    total_corrects = 0

    for x_batch, y_batch in tqdm(valid_loader, desc="Valid", leave=False, ncols=0):
        nll, corrects = valid_step(params, x_batch, y_batch)
        total_nll += nll.item() * x_batch.shape[0]
        total_corrects += corrects.item()

    valid_nll = total_nll / valid_loader.num_data
    valid_acc = total_corrects * 100 / valid_loader.num_data
    return valid_nll, valid_acc


def main(args):
    # Dataset
    dataset = get_train_dataset(
        name=args.data_name, root=args.data_root,
        num_data=args.num_data, valid_prop=args.valid_prop,
        normalize=True, onehot=True, seed=args.seed,
    )

    (x_train, y_train), (x_valid, y_valid), (num_class, data_name, data_msg) = dataset
    num_train = x_train.shape[0]
    num_valid = x_valid.shape[0]

    # Log and checkpoint
    if not args.ckpt_name:
        args.ckpt_name = f"{data_name}"
        args.ckpt_name += f"/{args.method}-{args.network}"
        args.ckpt_name += f"/nh{args.num_hiddens}-nc{args.num_channels}"
        if args.method == "tp":
            args.ckpt_name += f"-a{args.alpha:.0f}-b{args.beta:.0f}"
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
            num_class=num_class, num_train=num_train, num_valid=num_valid,
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

        def train_log(i, nll, log=True):
            print_str = f"NLL: {nll:.6f}"
            if log:
                logger.log(f"       [{i:4d}]  {print_str}", is_tqdm=True)
            return print_str

        # Train
        train_step = build_train_step(apply_fn, get_params, opt_update)
        valid_step = build_valid_step(apply_fn)
        train_loader = DataLoader(x_train, y_train, batch_size=args.num_batch, shuffle=True)
        valid_loader = DataLoader(x_valid, y_valid, batch_size=args.num_batch, shuffle=False)

        valid_nll, valid_acc = valid_epoch(params, valid_loader, valid_step)
        logger.log(f"[{0:3d}]  Valid NLL: {valid_nll:.6f}  Valid ACC: {valid_acc:.2f}")

        best_epoch, best_nll, best_acc = 0, valid_nll, valid_acc
        best_print_str = train_log(0, valid_nll, log=False)

        for epoch in trange(1, args.max_epoch + 1, desc="Epoch", ncols=0):
            train_nll, opt_state = train_epoch(epoch, opt_state, train_loader, train_step, train_log)
            logger.log(f"[{epoch:3d}]  Train NLL: {train_nll:.6f}", is_tqdm=True)

            valid_nll, valid_acc = valid_epoch(get_params(opt_state), valid_loader, valid_step)
            logger.log(f"[{epoch:3d}]  Valid NLL: {valid_nll:.6f}  Valid ACC: {valid_acc:.2f}", is_tqdm=True)

            if valid_nll < best_nll:
                best_epoch, best_nll, best_acc = epoch, valid_nll, valid_acc
                best_print_str = train_log(epoch, valid_nll, log=False)
                jnp.save(os.path.join(ckpt_dir, f"{epoch:03d}.npy"), (get_params(opt_state), list(net_kwargs.values())))
                logger.log(f"[{epoch:3d}]  Updated  NLL: {valid_nll:.6f}  ACC: {valid_acc:.2f}", is_tqdm=True)

        logger.log(f"[{best_epoch:3d}]  Valid NLL: {best_nll:.6f}  Valid ACC: {best_acc:.2f}  {best_print_str}")

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    except:
        import traceback
        logger.log(f"\n{traceback.format_exc()}\nStopped")

    finally:
        logger.close()
