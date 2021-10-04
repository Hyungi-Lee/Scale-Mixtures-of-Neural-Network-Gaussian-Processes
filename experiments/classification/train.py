import os
from datetime import datetime

import numpy as np
from tqdm import tqdm, trange

import jax
from jax import random

import objax
from objax import VarCollection
from objax.optimizer import SGD, Adam

from spax.models import SVSP
from spax.kernels import NNGPKernel
from spax.priors import GaussianPrior, InverseGammaPrior

from .data import get_train_dataset
from ..nt_kernels import get_cnn_kernel, get_conv_resnet_kernel
from ..utils import DataLoader, Checkpointer, Logger, ReduceLROnPlateau, get_context_summary


def add_subparser(subparsers):
    parser = subparsers.add_parser("train", aliases=["tr"])
    parser.set_defaults(func=main)

    parser.add_argument("-m",   "--method",           choices=["svgp", "svtp"], required=True)
    parser.add_argument("-n",   "--network",          choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("-dr",  "--data-root",        type=str, default="./data")
    parser.add_argument("-dn",  "--data-name",        required=True)
    parser.add_argument("-cr",  "--ckpt-root",        type=str, default="./_ckpt/cls")
    parser.add_argument("-cn",  "--ckpt-name",        type=str, default=None)

    parser.add_argument("-vp",  "--valid-prop",       type=float, default=0.1)
    parser.add_argument("-nd",  "--num-data",         type=int, default=None)
    parser.add_argument("-nb",  "--num-batch",        type=int, default=100)
    parser.add_argument("-ni",  "--num-inducing",     type=int, default=200)
    parser.add_argument("-ns",  "--num-sample",       type=int, default=100)
    parser.add_argument("-nvs", "--num-valid-sample", type=int, default=1000)

    parser.add_argument("-a",   "--alpha",            type=float, default=2.)
    parser.add_argument("-b",   "--beta",             type=float, default=2.)

    parser.add_argument("-nh",  "--num-hiddens",      type=int, default=4)
    parser.add_argument("-act", "--activation",       choices=["erf", "relu"], default="relu")
    parser.add_argument("-ws",  "--w-std",            type=float, default=1.)
    parser.add_argument("-bs",  "--b-std",            type=float, default=1e-8)
    parser.add_argument("-ls",  "--last-w-std",       type=float, default=1.)
    parser.add_argument("-eps", "--epsilon",          type=float, default=1e-6)

    parser.add_argument("-opt", "--optimizer",        choices=["adam", "sgd"], default="adam")
    parser.add_argument("-lr",  "--lr",               type=float, default=1e-2)
    parser.add_argument("-lr2", "--lr2",              type=float, default=None)
    parser.add_argument("-lrd", "--lr-decay",         type=float, default=0.5)
    parser.add_argument("-lrt", "--lr-threshold",     type=float, default=1e-4)
    parser.add_argument("-lrp", "--lr-patience",      type=int, default=5)
    parser.add_argument("-e",   "--max-epoch",        type=int, default=300)
    parser.add_argument("-r",   "--resize",           type=int, default=1)

    parser.add_argument("-s",   "--seed",             type=int, default=10)
    parser.add_argument("-q",   "--quite",            default=False, action="store_true")
    parser.add_argument("-c",   "--comment",          type=str, default="")


def build_train_step(model, train_vars, optimizer, num_train, num_samples, jit=True):
    grad_loss = objax.GradValues(model.loss, train_vars)
    def train_step(key, x_batch, y_batch, learning_rate):
        g, v = grad_loss(key, x_batch, y_batch, num_train, num_samples)
        optimizer(learning_rate, g)
        return v[0]
    return objax.Jit(train_step, grad_loss.vars() + optimizer.vars()) if jit else train_step


def build_train_step2(model, train_vars1, train_vars2, opt1, opt2, opt_idx1, num_train, num_samples, jit=True):
    grad_loss = objax.GradValues(model.loss, train_vars1 + train_vars2)
    def train_step2(key, x_batch, y_batch, lr1, lr2):
        g, v = grad_loss(key, x_batch, y_batch, num_train, num_samples)
        opt1(lr1, [v for i, v in enumerate(g) if i in opt_idx1])
        opt2(lr2, [v for i, v in enumerate(g) if i not in opt_idx1])
        return v[0]
    return objax.Jit(train_step2, grad_loss.vars() + opt1.vars("opt1") + opt2.vars("opt2")) if jit else train_step2


def build_valid_step(model, num_samples, jit=True):
    def valid_step(key, x_batch, y_batch):
        nll, correct_count = model.test_acc_nll(key, x_batch, y_batch, num_samples)
        return nll, correct_count
    return objax.Jit(valid_step, model.vars()) if jit else valid_step


def train_epoch(key, train_loader, train_step, learning_rate, train_log, learning_rate2=None):
    total_nelbo = 0.
    lenb = len(train_loader)
    log_interval = len(train_loader) // 4

    for idx, (x_batch, y_batch) in tqdm(enumerate(train_loader), desc="Train", leave=False, ncols=0, total=lenb):
        key, split_key = random.split(key)
        if learning_rate2 is None:
            nelbo = train_step(split_key, x_batch, y_batch, learning_rate)
        else:
            nelbo = train_step(split_key, x_batch, y_batch, learning_rate, learning_rate2)
        total_nelbo += nelbo.item() * x_batch.shape[0]
        if (idx + 1) % log_interval == 0:
            train_log(idx + 1, nelbo)

    train_nelbo = total_nelbo / train_loader.num_data
    return train_nelbo


def valid_epoch(key, valid_loader, valid_step):
    total_nll = 0.
    total_corrects = 0

    for x_batch, y_batch in tqdm(valid_loader, desc="Valid", leave=False, ncols=0):
        key, split_key = random.split(key)
        nll, corrects = valid_step(split_key, x_batch, y_batch)
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
        normalize=True, seed=args.seed,
    )

    (x_train, y_train), (x_valid, y_valid), (num_class, data_name, data_msg) = dataset
    num_train = x_train.shape[0]
    num_valid = x_valid.shape[0]

    # Log and checkpoint
    if not args.ckpt_name:
        args.ckpt_name = f"{data_name}"
        args.ckpt_name += f"/{args.method}-{args.network}"
        args.ckpt_name += f"/ni{args.num_inducing}-nh{args.num_hiddens}"
        if args.method == "svtp":
            args.ckpt_name += f"-a{args.alpha:.1f}-b{args.beta:.1f}"
        if args.comment:
            args.ckpt_name += f"/{args.comment}"
        else:
            args.ckpt_name += f"/{str(datetime.now().strftime('%y%m%d%H%M'))}"

    ckpt_dir = os.path.join(os.path.expanduser(args.ckpt_root), args.ckpt_name)
    checkpointer = Checkpointer(ckpt_dir, keep_ckpts=20)
    logger = Logger(ckpt_dir, quite=args.quite)

    try:
        # Resize
        h, w, c = x_train.shape[1:]

        if args.resize > 1:
            new_h, new_w = h // args.resize, w // args.resize
            x_train = jax.image.resize(x_train, (num_train, new_h, new_w, c), method="bilinear")
            x_valid = jax.image.resize(x_valid, (num_valid, new_h, new_w, c), method="bilinear")
            logger.log(f"Resized to ({h}, {w}, {c}) -> ({new_h}, {new_w}, {c})")

        # Kernel
        if args.network is None or args.network == "cnn":
            args.network = "cnn"
            base_kernel_fn = get_cnn_kernel
        else:
            args.network = "resnet"
            base_kernel_fn = get_conv_resnet_kernel

        def get_kernel_fn(w_std, b_std, last_w_std):
            kernel_fn = base_kernel_fn(
                args.num_hiddens, num_class, args.activation,
                w_std=w_std, b_std=b_std, last_w_std=last_w_std,
            )
            return kernel_fn

        if args.method == "svgp":
            kernel = NNGPKernel(get_kernel_fn, args.w_std, args.b_std, args.last_w_std)
        elif args.method == "svtp":
            kernel = NNGPKernel(get_kernel_fn, args.w_std, args.b_std, 1.)

        # Model

        ## Inducing points
        label_class = np.array([sum(y_train == class_idx) for class_idx in range(num_class)])
        num_inducing_class = np.round(args.num_inducing * label_class / sum(label_class)).astype(int).tolist()
        inducing_points = np.concatenate([x_train[y_train == class_idx][:ni]
                                          for class_idx, ni in zip(range(num_class), num_inducing_class)], axis=0)
        args.num_inducing = inducing_points.shape[0]

        ## Prior
        if args.method == "svgp":
            prior = GaussianPrior()
        elif args.method == "svtp":
            prior = InverseGammaPrior(args.alpha, args.beta)
        else:
            raise ValueError(f"Unsupported method '{args.method}'")

        ## Model
        model = SVSP(prior, kernel, inducing_points, num_latent_gps=num_class, eps=args.epsilon)
        model_vars = model.vars()

        if args.method == "svgp":
            train_vars = model.vars()

            def train_log(i, nelbo, log=True):
                ws, bs, ls = model.kernel.get_params()
                eps = model.eps.safe_value
                print_str = f"nELBO: {nelbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}  e: {eps:.3E}"
                if log:
                    logger.log(f"       [{i:4d}]  {print_str}", is_tqdm=True)
                return print_str

        elif args.method == "svtp":
            train_vars = VarCollection({k: v for k, v in model.vars().items() if "last_w_std" not in k})

            def train_log(i, nelbo, log=True):
                ws, bs, _ = model.kernel.get_params()
                eps = model.eps.safe_value
                ia, ib = model.prior.a.safe_value, model.prior.b.safe_value
                print_str = f"nELBO: {nelbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  a: {ia:.4f}  b: {ib:.4f}  e: {eps:.3E}"
                if log:
                    logger.log(f"       [{i:4d}]  {print_str}", is_tqdm=True)
                return print_str

        # Optimizer
        if args.lr2:
            train_vars1 = VarCollection({k: v for k, v in model.vars().items() if "prior" not in k})
            train_vars2 = VarCollection({k: v for k, v in model.vars().items() if "prior" in k})
            opt_idx1 = [i for i, k in enumerate((train_vars1 + train_vars2).keys()) if "prior" not in k]
            if args.optimizer == "adam":
                optimizer1 = Adam(train_vars1)
                optimizer2 = Adam(train_vars2)
            elif args.optimizer == "sgd":
                optimizer1 = SGD(train_vars1)
                optimizer2 = SGD(train_vars2)
        else:
            if args.optimizer == "adam":
                optimizer = Adam(train_vars)
            elif args.optimizer == "sgd":
                optimizer = SGD(train_vars)

        scheduler = ReduceLROnPlateau(lr=args.lr, factor=args.lr_decay, patience=args.lr_patience)

        # Log
        np.save(os.path.join(ckpt_dir, "meta.npy"), vars(args))
        logger.log(get_context_summary(args, dict(
            num_class=num_class, num_train=num_train, num_valid=num_valid,
            data_name=data_name, data_msg=data_msg,
            num_inducing=args.num_inducing, inducing_points=num_inducing_class,
        )))

        # Build functions
        train_loader = DataLoader(x_train, y_train, batch_size=args.num_batch, shuffle=True, seed=args.seed)
        valid_loader = DataLoader(x_valid, y_valid, batch_size=args.num_batch, shuffle=False)

        if args.lr2:
            train_step = build_train_step2(model, train_vars1, train_vars2, optimizer1, optimizer2, opt_idx1, num_train, args.num_sample)
        else:
            train_step = build_train_step(model, train_vars, optimizer, num_train, args.num_sample)
        valid_step = build_valid_step(model, args.num_valid_sample)

        # Train
        key = random.PRNGKey(args.seed)

        valid_nll, valid_acc = valid_epoch(key, valid_loader, valid_step)
        logger.log(f"[{0:3d}]  NLL: {valid_nll:.5f}  ACC: {valid_acc:.2f}")

        best_epoch, best_nll, best_acc, best_print_str = 0, valid_nll, valid_acc, ""
        _ = checkpointer.step(0, valid_nll, model_vars)

        for epoch in trange(1, args.max_epoch + 1, desc="Epoch", ncols=0):
            key, split_key = random.split(key)

            train_nelbo = train_epoch(split_key, train_loader, train_step, scheduler.lr, train_log, args.lr2)
            logger.log(f"[{epoch:3d}]  nELBO: {train_nelbo:.5f}", is_tqdm=True)

            valid_nll, valid_acc = valid_epoch(split_key, valid_loader, valid_step)
            logger.log(f"[{epoch:3d}]  NLL: {valid_nll:.5f}  ACC: {valid_acc:.2f}", is_tqdm=True)

            updated = checkpointer.step(epoch, valid_nll, model_vars)
            if updated:
                best_epoch, best_nll, best_acc = epoch, valid_nll, valid_acc
                best_print_str = train_log(epoch, train_nelbo, log=False)
                logger.log(f"[{epoch:3d}]  Updated  NLL: {valid_nll:.5f}  ACC: {valid_acc:.2f}", is_tqdm=True)

            reduced = scheduler.step(valid_nll)
            if reduced:
                logger.log(f"[{epoch:3d}]  LR reduced to {scheduler.lr:.6f}", is_tqdm=True)
                if scheduler.lr < args.lr_threshold:
                    break

        logger.log(f"[{best_epoch:3d}]  NLL: {best_nll:.5f}  ACC: {best_acc:.2f}  {best_print_str}")

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    except:
        import traceback
        logger.log(f"\n{traceback.format_exc()}\nStopped")

    finally:
        logger.close()
