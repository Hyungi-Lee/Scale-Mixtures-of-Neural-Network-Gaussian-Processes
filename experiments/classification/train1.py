import os
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd

from jax import random
from jax import numpy as jnp

import objax
from objax import VarCollection
from objax.optimizer import SGD, Adam

from spax.models import SVSP
from spax.kernels import NNGPKernel
from spax.priors import GaussianPrior, InverseGammaPrior

from .data import get_train_dataset, datasets
from ..nt_kernels import get_mlp_kernel, get_cnn_kernel, get_resnet_kernel
from ..utils import TrainBatch, TestBatch, Checkpointer, Logger, ReduceLROnPlateau, get_context_summary


def add_subparser(subparsers):
    parser = subparsers.add_parser("train1", aliases=["tr1"])
    parser.set_defaults(func=main)

    parser.add_argument("-m",   "--method",           choices=["svgp", "svtp"], required=True)
    parser.add_argument("-n",   "--network",          choices=["cnn", "resnet", "mlp"], default=None)
    parser.add_argument("-dn",  "--data-name",        choices=datasets, required=True)
    parser.add_argument("-dr",  "--data-root",        type=str, default="./data")
    parser.add_argument("-cr",  "--ckpt-root",        type=str, default="./_ckpt")
    parser.add_argument("-cn",  "--ckpt-name",        type=str, default=None)

    parser.add_argument("-vp",  "--valid-prop",       type=float, default=0.1)
    parser.add_argument("-nd",  "--num-data",         type=int, default=None)
    parser.add_argument("-nb",  "--num-batch",        type=int, default=128)
    parser.add_argument("-ni",  "--num-inducing",     type=int, default=200)
    parser.add_argument("-ns",  "--num-sample",       type=int, default=100)
    parser.add_argument("-nvs", "--num-valid-sample", type=int, default=1000)

    parser.add_argument("-a",   "--alpha",            type=float, default=2.)
    parser.add_argument("-b",   "--beta",             type=float, default=2.)
    parser.add_argument("-al",  "--alphas",           type=str, default="[.5]")
    parser.add_argument("-bl",  "--betas",            type=str, default="[.5]")

    parser.add_argument("-nh",  "--num-hiddens",      type=int, default=4)
    parser.add_argument("-act", "--activation",       choices=["erf", "relu"], default="relu")
    parser.add_argument("-ws",  "--w-std",            type=float, default=1.)
    parser.add_argument("-bs",  "--b-std",            type=float, default=1e-8)
    parser.add_argument("-ls",  "--last-w-std",       type=float, default=1.)

    parser.add_argument("-opt", "--optimizer",        choices=["adam", "sgd"], default="adam")
    parser.add_argument("-lr1", "--lr1",              type=float, default=1e-2)
    parser.add_argument("-lr2", "--lr2",              type=float, default=1e-2)
    parser.add_argument("-lrd", "--lr-decay",         type=float, default=0.5)
    parser.add_argument("-lrt", "--lr-threshold",     type=float, default=1e-4)
    parser.add_argument("-lrp", "--lr-patience",      type=int, default=5)
    parser.add_argument("-t",   "--max-steps",        type=int, default=30000)
    parser.add_argument("-km",  "--kmeans",           default=False, action="store_true")

    parser.add_argument("-s",   "--seed",             type=int, default=10)
    parser.add_argument("-pi",  "--print-interval",   type=int, default=100)
    parser.add_argument("-vi",  "--valid-interval",   type=int, default=500)
    parser.add_argument("-q",   "--quite",            default=False, action="store_true")
    parser.add_argument("-c",   "--comment",          type=str, default="")


def indent(string, n=8):
    return "\n".join(map(lambda s: " " * n + s, str(string).split("\n")))


def get_inducing_points(X, Y, num_inducing, num_class):
    inducings = []
    inducing_per_class = num_inducing // num_class
    for class_idx in range(num_class):
        xc = X[jnp.argmax(Y, axis=1) == class_idx]
        xc = xc.reshape(xc.shape[0], -1)
        kmeans = KMeans(n_clusters=inducing_per_class).fit(xc).cluster_centers_
        inducings.append(kmeans.reshape(kmeans.shape[0], *X.shape[1:]))
    return jnp.vstack(inducings)


def build_train_step(model, opt1, opt2, opt1_idxs, opt2_idxs, num_train, num_samples, jit=True):
    grad_loss = objax.GradValues(model.loss, model.vars())
    def train_step(key, x_batch, y_batch, lr1, lr2):
        g, v = grad_loss(key, x_batch, y_batch, num_train, num_samples)
        opt1(lr1, [v for i, v in enumerate(g) if i in opt1_idxs])
        opt2(lr2, [v for i, v in enumerate(g) if i in opt2_idxs])
        return v[0]
    return objax.Jit(train_step, grad_loss.vars() + opt1.vars("opt1") + opt2.vars("opt2")) if jit else train_step


def build_train_step2(model, num_train, num_samples, alphas, betas, jit=True):
    def train_step(key, x_batch, y_batch):
        v = model.loss2(key, x_batch, y_batch, num_train, num_samples, alphas, betas)
        return v
    return objax.Jit(train_step, model.vars()) if jit else train_step


def build_valid_step(model, num_samples, jit=True):
    def valid_step(key, x_batch, y_batch):
        nll, correct_count = model.test_acc_nll(key, x_batch, y_batch, num_samples)
        return nll, correct_count
    return objax.Jit(valid_step, model.vars()) if jit else valid_step


def validate(key, valid_batches, valid_step, num_valid):
    valid_nll_list = []
    total_corrects = 0

    for x_batch, y_batch in tqdm(valid_batches, desc="Valid", leave=False, ncols=0):
        key, split_key = random.split(key)
        nll, corrects = valid_step(split_key, x_batch, y_batch)
        valid_nll_list.append(nll)
        total_corrects += corrects

    valid_nll = (jnp.sum(jnp.array(valid_nll_list)) / num_valid).item()
    valid_acc = (total_corrects / num_valid).item()
    return valid_nll, valid_acc


def build_valid_step2(model, num_samples, alphas, betas, jit=True):
    def valid_step(key, x_batch, y_batch):
        nlls, ccs = model.test_acc_nll2(key, x_batch, y_batch, num_samples, alphas, betas)
        return nlls, ccs
    return objax.Jit(valid_step, model.vars()) if jit else valid_step


def validate2(key, valid_batches, valid_step, num_valid, alphas, betas):
    valid_nll_list = []
    total_corrects = np.zeros((len(alphas), len(betas)))

    for x_batch, y_batch in tqdm(valid_batches, desc="Valid", leave=False, ncols=0):
        key, split_key = random.split(key)
        nlls, ccs = valid_step(split_key, x_batch, y_batch)
        nlls = np.array(nlls).reshape(len(alphas), len(betas))
        ccs = np.array(ccs).reshape(len(alphas), len(betas))
        valid_nll_list.append(nlls)
        total_corrects += ccs

    valid_nll = (jnp.sum(jnp.array(valid_nll_list), axis=0) / num_valid)
    valid_acc = (total_corrects / num_valid)
    return valid_nll, valid_acc


def main(args):
    # Log
    if not args.ckpt_name:
        args.ckpt_name = f"{args.data_name}"
        args.ckpt_name += f"/{args.method}"
        args.ckpt_name += f"/ni{args.num_inducing}-nh{args.num_hiddens}"
        if args.method == "svtp":
            args.ckpt_name += f"-a{args.alpha:.1f}-b{args.beta:.1f}"
        if args.kmeans:
            args.ckpt_name += "-km"
        # args.ckpt_name += f"-s{args.seed}"
        if args.comment:
            args.ckpt_name += f"/{args.comment}"
        else:
            args.ckpt_name += f"/{str(datetime.now().strftime('%y%m%d%H%M'))}"

    ckpt_dir = os.path.join(os.path.expanduser(args.ckpt_root), args.ckpt_name)
    checkpointer = Checkpointer(ckpt_dir, patience=15)
    logger = Logger(ckpt_dir, quite=args.quite)

    alphas = eval(args.alphas)
    betas = eval(args.betas)

    try:
        # Dataset
        dataset = get_train_dataset(
            name=args.data_name, root=args.data_root,
            num_data=args.num_data, valid_prop=args.valid_prop,
            normalize=True, seed=args.seed,
        )
        x_train, y_train, x_valid, y_valid, dataset_info = dataset

        num_class = dataset_info["num_class"]
        num_train = x_train.shape[0]
        num_valid = x_valid.shape[0]

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
            kernel_fn = base_kernel_fn(
                args.num_hiddens, num_class, args.activation,
                w_std=w_std, b_std=b_std, last_w_std=last_w_std,
            )
            return kernel_fn

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

        # Optimizer
        idx1 = [i for i, k in enumerate(model.vars().keys()) if "prior" not in k]
        idx2 = [i for i, k in enumerate(model.vars().keys()) if "prior" in k]
        vars1 = VarCollection({k: v for k, v in model.vars().items() if "prior" not in k})
        vars2 = VarCollection({k: v for k, v in model.vars().items() if "prior" in k})
        if args.optimizer == "adam":
            opt1 = Adam(vars1)
            opt2 = Adam(vars2)
        elif args.optimizer == "sgd":
            opt1 = SGD(vars1)
            opt2 = SGD(vars2)
        else:
            raise ValueError(f"Unsupported optimizer '{args.optimizer}'")

        scheduler = ReduceLROnPlateau(lr=args.lr1, factor=args.lr_decay, patience=args.lr_patience)

        # Build functions
        num_valid_batch = 100

        train_step = build_train_step(model, opt1, opt2, idx1, idx2, num_train, args.num_sample)
        train_step2 = build_train_step2(model, num_train, args.num_sample, alphas, betas)
        valid_step = build_valid_step(model, args.num_valid_sample)
        valid_step2 = build_valid_step2(model, args.num_valid_sample, alphas, betas)

        train_batches = TrainBatch(x_train, y_train, args.num_batch, args.seed)
        valid_batches = TestBatch(x_valid, y_valid, num_valid_batch)
        valid_train_batches = TestBatch(x_train, y_train, num_valid_batch)

        save_vc = model.vars() + opt1.vars("opt1") + opt2.vars("opt2")

        # Log
        np.save(os.path.join(ckpt_dir, "meta.npy"), dict(dataset=dataset_info, args=vars(args)))
        logger.log(get_context_summary(args, dict(num_class=num_class, num_train=num_train, num_valid=num_valid)))

        # Train
        key = random.PRNGKey(args.seed)

        # train_nll, train_acc = validate2(key, valid_train_batches, valid_step2, num_train, alphas, betas)
        valid_nll, valid_acc = validate2(key, valid_batches, valid_step2, num_valid, alphas, betas)
        # logger.log(f"[{0:5d}] Train " + "=" * 40)
        # logger.log(indent(pd.DataFrame(train_nll, index=alphas, columns=betas)))
        # with pd.option_context("display.float_format", "{:.2f}".format):
        #     logger.log(indent(pd.DataFrame(train_acc * 100, index=alphas, columns=betas)))
        logger.log(f"[{0:5d}] Valid " + "=" * 40)
        logger.log(indent(pd.DataFrame(valid_nll, index=alphas, columns=betas)))
        with pd.option_context("display.float_format", "{:.2f}".format):
            logger.log(indent(pd.DataFrame(valid_acc * 100, index=alphas, columns=betas)))

        best_model_nll, best_model_acc = valid_nll, valid_acc
        best_step = 0
        best_print_str = ""

        for i, (x_batch, y_batch) in tqdm(enumerate(train_batches, start=1), total=args.max_steps, desc="Train", ncols=0):
            key, split_key = random.split(key)
            n_elbo = train_step(split_key, x_batch, y_batch, scheduler.lr, args.lr2)

            if i % args.print_interval == 0:
                ws, bs, ls = model.kernel.get_params()

                if args.method == "svtp":
                    ia, ib = (model.prior.a.safe_value, model.prior.b.safe_value)
                    print_str = f"nELBO: {n_elbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}  a: {ia:.4f}  b: {ib:.4f}"
                else:
                    print_str = f"nELBO: {n_elbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}"

                logger.log(f"[{i:5d}] {print_str}", is_tqdm=True)

            if i % args.valid_interval == 0:
                valid_nll, valid_acc = validate(key, valid_batches, valid_step, num_valid)
                logger.log(f"[{i:5d}] NLL: {valid_nll:.5f}  ACC: {valid_acc:.4f}", is_tqdm=True)
                reduced = scheduler.step(valid_nll)
                updated, _ = checkpointer.step(i, valid_nll, save_vc)

                if updated:
                    logger.log(f"[{i:5d}] Updated  NLL: {valid_nll:.5f}  ACC: {valid_acc*100:.2f}", is_tqdm=True)
                    best_step, best_model_acc, best_model_nll = i, valid_acc, valid_nll
                    best_print_str = print_str

                if reduced:
                    logger.log(f"LR reduced to {scheduler.lr:.6f}", is_tqdm=True)
                    if scheduler.lr < args.lr_threshold:
                        break

                # train_nll, train_acc = validate2(key, valid_train_batches, valid_step2, num_train, alphas, betas)
                nelbos = np.array(train_step2(split_key, x_batch, y_batch)).reshape(len(alphas), len(betas))
                valid_nll, valid_acc = validate2(key, valid_batches, valid_step2, num_valid, alphas, betas)
                logger.log(f"[{i:5d}] Train " + "=" * 40, is_tqdm=True)
                # logger.log(indent(pd.DataFrame(train_nll, index=alphas, columns=betas)), is_tqdm=True)
                # with pd.option_context("display.float_format", "{:.2f}".format):
                    # logger.log(indent(pd.DataFrame(train_acc * 100, index=alphas, columns=betas)), is_tqdm=True)
                logger.log(indent(pd.DataFrame(nelbos, index=alphas, columns=betas)), is_tqdm=True)
                logger.log(f"[{i:5d}] Valid " + "=" * 40, is_tqdm=True)
                logger.log(indent(pd.DataFrame(valid_nll, index=alphas, columns=betas)), is_tqdm=True)
                with pd.option_context("display.float_format", "{:.2f}".format):
                    logger.log(indent(pd.DataFrame(valid_acc * 100, index=alphas, columns=betas)), is_tqdm=True)

            if i == args.max_steps:
                break

        logger.log(f"\n[{best_step:5d}] NLL: {best_model_nll:.5f}  ACC: {best_model_acc:.4f}  {best_print_str}\n")

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    except:
        import traceback
        logger.log(f"\n{traceback.format_exc()}\nStopped")

    finally:
        logger.close()
