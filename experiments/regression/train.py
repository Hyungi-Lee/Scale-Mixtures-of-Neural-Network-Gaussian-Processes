import os
import math
from datetime import datetime

import numpy as np
from tqdm import tqdm

from jax import numpy as jnp

import objax
from objax import VarCollection
from objax.optimizer import SGD, Adam

from spax.models import SPR
from spax.kernels import NNGPKernel
from spax.likelihoods import GaussianLikelihood, StudentTLikelihood

from .data import DATASETS, get_dataset, permute_dataset, split_dataset
from ..nt_kernels import get_mlp_kernel, get_dense_resnet_kernel
from ..utils import Checkpointer, Logger, ReduceLROnPlateau, get_context_summary


def add_subparser(subparsers):
    parser = subparsers.add_parser("train", aliases=["tr"])
    parser.set_defaults(func=main)

    parser.add_argument("-m",   "--method",           choices=["gp", "tp"], required=True)
    parser.add_argument("-n",   "--network",          choices=["resnet", "mlp"], default=None)
    parser.add_argument("-dn",  "--data-name",        choices=DATASETS, required=True)
    parser.add_argument("-dr",  "--data-root",        type=str, default="./data")
    parser.add_argument("-cr",  "--ckpt-root",        type=str, default="./_ckpt")
    parser.add_argument("-cn",  "--ckpt-name",        type=str, default=None)

    parser.add_argument("-vp",  "--valid-prop",       type=float, default=0.1)
    parser.add_argument("-nd",  "--num-data",         type=int, default=None)

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
    parser.add_argument("-lrd", "--lr-decay",         type=float, default=0.5)
    parser.add_argument("-lrt", "--lr-threshold",     type=float, default=1e-4)
    parser.add_argument("-lrp", "--lr-patience",      type=int, default=5)
    parser.add_argument("-t",   "--max-steps",        type=int, default=30000)

    parser.add_argument("-s",   "--seed",             type=int, default=10)
    parser.add_argument("-pi",  "--print-interval",   type=int, default=100)
    parser.add_argument("-vi",  "--valid-interval",   type=int, default=500)
    parser.add_argument("-q",   "--quite",            default=False, action="store_true")
    parser.add_argument("-c",   "--comment",          type=str, default="")


def build_train_step(model, train_vars, optimizer, jit=True):
    grad_loss = objax.GradValues(model.loss, train_vars)
    def train_step(learning_rate):
        g, v = grad_loss()
        optimizer(learning_rate, g)
        return v[0]
    return objax.Jit(train_step, grad_loss.vars() + optimizer.vars()) if jit else train_step


def build_valid_step(model, x_valid, y_valid, jit=True):
    def valid_step():
        nll = model.test_nll(x_valid, y_valid)
        return nll
    return objax.Jit(valid_step, model.vars()) if jit else valid_step


def main(args):
    # Log
    if not args.ckpt_name:
        args.ckpt_name = f"{args.data_name}"
        args.ckpt_name += f"/{args.method}"
        args.ckpt_name += f"/nh{args.num_hiddens}-ws{args.w_std:.1f}-bs{args.b_std:.1f}-ls{args.last_w_std:.1f}"
        if args.method == "tp":
            args.ckpt_name += f"-a{args.alpha:.1f}-b{args.beta:.1f}"
        if args.comment:
            args.ckpt_name += f"/{args.comment}"
        else:
            args.ckpt_name += f"/{str(datetime.now().strftime('%y%m%d%H%M'))}"

    ckpt_dir = os.path.join(os.path.expanduser(args.ckpt_root), args.ckpt_name)
    checkpointer = Checkpointer(ckpt_dir)
    logger = Logger(ckpt_dir, quite=args.quite)

    try:
        # Dataset
        x, y = get_dataset(name=args.data_name, root=args.data_root)
        x, y = permute_dataset(x, y, seed=10)
        splits = split_dataset(x, y, train=0.8, valid=0.1, test=0.1)
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (y_std, y_mean) = splits

        num_train = x_train.shape[0]
        num_valid = x_valid.shape[0]

        x_train_valid = np.concatenate([x_train, x_valid], axis=0)
        y_train_valid = np.concatenate([y_train, y_valid], axis=0)

        x_train_valid, y_train_valid = permute_dataset(x_train_valid, y_train_valid, seed=args.seed)
        x_train, x_valid = x_train_valid[:num_train], x_train_valid[num_train:]
        y_train, y_valid = y_train_valid[:num_train], y_train_valid[num_train:]

        x_train, y_train = jnp.array(x_train), jnp.array(y_train)
        x_valid, y_valid = jnp.array(x_valid), jnp.array(y_valid)
        x_test, y_test = jnp.array(x_test), jnp.array(y_test)
        y_std, y_mean = jnp.array(y_std), jnp.array(y_mean)

        # Model
        if args.network is None or args.network == "mlp":
            args.network = "mlp"
            base_kernel_fn = get_mlp_kernel
        elif args.network == "resnet":
            args.network = "resnet"
            base_kernel_fn = get_dense_resnet_kernel
        else:
            raise ValueError(f"Unsupported network '{args.network}'")

        def get_kernel_fn(w_std, b_std, last_w_std):
            kernel_fn = base_kernel_fn(
                args.num_hiddens, act=args.activation,
                w_std=w_std, b_std=b_std, last_w_std=last_w_std,
            )
            return kernel_fn

        if args.method == "gp":
            kernel = NNGPKernel(get_kernel_fn, args.w_std, args.b_std, args.last_w_std)
            likelihood = GaussianLikelihood()

        elif args.method == "tp":
            # kernel = NNGPKernel(get_kernel_fn, args.w_std, args.b_std, 1.)
            kernel = NNGPKernel(get_kernel_fn, args.w_std, args.b_std, args.last_w_std)   # TODO: check
            likelihood = StudentTLikelihood(args.alpha, args.beta)

        model = SPR(kernel, likelihood, x_train, y_train, y_mean, y_std, eps=args.epsilon)
        model_vars = model.vars()

        if args.method == "gp":
            train_vars = model.vars()
        elif args.method == "tp":
            # train_vars = VarCollection({k: v for k, v in model.vars().items() if "last_w_std" not in k})
            train_vars = model.vars()  # TODO: check

        # Optimizer
        if args.optimizer == "adam":
            optimizer = Adam(train_vars)
        elif args.optimizer == "sgd":
            optimizer = SGD(train_vars)
        else:
            raise ValueError(f"Unsupported optimizer '{args.optimizer}'")

        scheduler = ReduceLROnPlateau(lr=args.lr, factor=args.lr_decay, patience=args.lr_patience)

        # Build functions
        train_step = build_train_step(model, train_vars, optimizer)
        valid_step = build_valid_step(model, x_valid, y_valid)
        test_step = build_valid_step(model, x_test, y_test)

        # Log
        np.save(os.path.join(ckpt_dir, "meta.npy"), dict(args=vars(args)))
        logger.log(get_context_summary(args, dict(num_train=num_train, num_valid=num_valid)))

        # Train
        valid_nll = valid_step()
        test_nll = test_step()
        logger.log(f"[{0:5d}] NLL: {valid_nll:.5f}  TEST: {test_nll:.5f}")

        best_step, best_nll, best_test_nll, best_print_str = 0, valid_nll, test_nll, ""
        _ = checkpointer.step(0, valid_nll, model_vars)

        for i in tqdm(range(1, args.max_steps + 1), desc="Train", ncols=0):
            nll = train_step(scheduler.lr)

            if i % args.print_interval == 0:
                ws, bs, ls = model.kernel.get_params()
                eps = model.eps.safe_value

                if args.method == "tp":
                    ia, ib = (model.likelihood.a.safe_value, model.likelihood.b.safe_value)
                    # print_str = f"nll: {nll:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  a: {ia:.4f}  b: {ib:.4f}  e: {eps:.3E}"  # TODO: check
                    print_str = f"nll: {nll:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}  a: {ia:.4f}  b: {ib:.4f}  e: {eps:.3E}"
                else:
                    print_str = f"nll: {nll:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}  e: {eps:.3E}"

                logger.log(f"[{i:5d}] {print_str}", is_tqdm=True)

            if i % args.valid_interval == 0:
                valid_nll = valid_step()
                test_nll = test_step()
                logger.log(f"[{i:5d}] NLL: {valid_nll:.5f}  TEST: {test_nll:.5f}", is_tqdm=True)
                reduced = scheduler.step(valid_nll)
                updated = checkpointer.step(i, valid_nll, model_vars)

                if updated:
                    logger.log(f"[{i:5d}] Updated  NLL: {valid_nll:.5f}  TEST: {test_nll:.5f}", is_tqdm=True)
                    best_step, best_nll, best_test_nll = i, valid_nll, test_nll
                    best_print_str = print_str

                if reduced:
                    logger.log(f"LR reduced to {scheduler.lr:.6f}", is_tqdm=True)
                    if scheduler.lr < args.lr_threshold:
                        break

                if math.isnan(valid_nll):
                    break

        logger.log(f"\n[{best_step:5d}] NLL: {best_nll:.5f}  TEST: {best_test_nll:.5f}  {best_print_str}\n")

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    except:
        import traceback
        logger.log(f"\n{traceback.format_exc()}\nStopped")

    finally:
        logger.close()
