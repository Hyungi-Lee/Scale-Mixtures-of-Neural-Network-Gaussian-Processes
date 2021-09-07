from tqdm import tqdm
from sklearn.cluster import KMeans

from jax import jit, random
from jax import numpy as jnp

import objax
from objax.optimizer import SGD, Adam

from neural_tangents import stax

from spax.models import SVSP
from spax.kernels import NNGPKernel, NNGPKernelParams
from spax.priors import GaussianPrior, InverseGammaPrior

from .data.classification import get_dataset, datasets
from .utils import TrainBatch, TestBatch


def add_subparser(subparsers):
    parser = subparsers.add_parser("classification", aliases=["cls"])
    parser.set_defaults(func=main)

    parser.add_argument("-m",   "--method",            choices=["svgp", "svtp"], required=True)
    parser.add_argument("-dn",  "--data-name",         choices=datasets, required=True)
    parser.add_argument("-tdn", "--test-data-name",    choices=datasets, default=None)
    parser.add_argument("-dr",  "--data-root",         type=str, default="./data")
    parser.add_argument("-lgr", "--log-root",          type=str, default="./_log")
    parser.add_argument("-ln",  "--log-name",          type=str)

    parser.add_argument("-ntr",  "--num-train",        type=int, default=None)
    parser.add_argument("-nts",  "--num-test",         type=int, default=None)
    parser.add_argument("-nb",   "--num-batch",        type=int, default=128)
    parser.add_argument("-ni",   "--num-induce",       type=int, default=100)
    parser.add_argument("-ntrs", "--num-train-sample", type=int, default=100)
    parser.add_argument("-ntss", "--num-test-sample",  type=int, default=10000)

    parser.add_argument("-a",    "--alpha",            type=float, default=2.)
    parser.add_argument("-b",    "--beta",             type=float, default=2.)

    parser.add_argument("-nh",   "--num-hiddens",      type=int, default=4)
    parser.add_argument("-act",  "--activation",       choices=["erf", "relu"], default="relu")
    parser.add_argument("-ws",   "--w-std",            type=float, default=1.)
    parser.add_argument("-bs",   "--b-std",            type=float, default=1e-8)
    parser.add_argument("-ls",   "--last-w-std",       type=float, default=1.)

    parser.add_argument("-opt",  "--optimizer",        choices=["adam", "sgd"], default="adam")
    parser.add_argument("-lr",   "--learning-rate",    type=float, default=1e-3)
    parser.add_argument("-t",    "--steps",            type=int, default=50000)
    parser.add_argument("-km",   "--kmeans",           default=False, action="store_true")

    parser.add_argument("-s",    "--seed",             type=int, default=10)
    parser.add_argument("-pi",   "--print-interval",   type=int, default=100)
    parser.add_argument("-ti",   "--test-interval",    type=int, default=500)


def get_inducing_points(X, Y, num_inducing, num_classes):
    inducings = []
    inducing_per_class = num_inducing // num_classes
    for class_idx in range(num_classes):
        xc = X[jnp.argmax(Y, axis=1) == class_idx]
        xc = xc.reshape(xc.shape[0], -1)
        kmeans = KMeans(n_clusters=inducing_per_class).fit(xc).cluster_centers_
        inducings.append(kmeans.reshape(kmeans.shape[0], *X.shape[1:]))
    return jnp.vstack(inducings)


def get_act_class(act):
    if act == "relu":
        return stax.Relu
    elif act == "erf":
        return stax.Erf
    else:
        raise KeyError("Unsupported act '{}'".format(act))


def get_cnn_kernel(num_hiddens, num_classes, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    layers = []
    for _ in range(num_hiddens):
        layers.append(stax.Conv(1, (3, 3), (1, 1), "SAME", W_std=w_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(num_classes, W_std=last_w_std))

    _, _, kernel_fn = stax.serial(*layers)
    # kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return  kernel_fn


def get_mlp_kernel(num_hiddens, num_classes, act="relu", w_std=1., b_std=0., last_w_std=1.):
    act_class = get_act_class(act)

    layers = []
    for _ in range(num_hiddens):
        layers.append(stax.Dense(512, W_std=w_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Dense(num_classes, W_std=last_w_std))

    _, _, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return  kernel_fn


def build_train_batch(model, optimizer, learning_rate, num_batch, num_train, num_samples, jit=True):
    grad_loss = objax.GradValues(model.loss, model.vars())
    def train_batch(key, x_batch, y_batch):
        g, v = grad_loss(key, x_batch, y_batch, num_batch, num_train, num_samples)
        optimizer(learning_rate, g)
        return v[0]
    return objax.Jit(train_batch, grad_loss.vars() + optimizer.vars()) if jit else train_batch


def build_test_batch(model, num_batch, num_samples, jit=True):
    def test_batch(key, x_batch, y_batch):
        nll, correct_count = model.test_acc_nll(key, x_batch, y_batch, num_batch, num_samples)
        return nll, correct_count
    return objax.Jit(test_batch, model.vars()) if jit else test_batch


def main(args):
    # Log
    # TODO

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

    print(f"{num_train = }, {num_test = }")

    # Kernel
    if dataset_info["type"] == "image":
        base_kernel_fn = get_cnn_kernel
    elif dataset_info["type"] == "feature":
        base_kernel_fn = get_mlp_kernel
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    def get_kernel_fn(w_std, b_std, last_w_std):
        return base_kernel_fn(
            args.num_hiddens, num_class, args.activation,
            w_std=w_std, b_std=b_std, last_w_std=last_w_std,
        )

    kernel = NNGPKernel(get_kernel_fn)
    kernel_params = NNGPKernelParams(args.w_std, args.b_std, args.last_w_std)

    # Model
    if args.kmeans:
        inducing_points = get_inducing_points(x_train, y_train, args.num_induce, num_class)
    else:
        inducing_points = x_train[:args.num_induce]

    if args.method == "svgp":
        prior = GaussianPrior()
    elif args.method == "svtp":
        prior = InverseGammaPrior(args.alpha, args.beta)
    else:
        raise ValueError(f"Unsupported method '{args.method}'")

    model = SVSP(prior, kernel, kernel_params, inducing_points, num_latent_gps=num_class)

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

    # Train
    # NOTE: Be careful to update kernel
    key = random.PRNGKey(args.seed)

    best_model_acc = 0.
    best_model_nll = float("inf")
    best_step = 0
    best_print_str = ""

    for i, (x_batch, y_batch) in tqdm(enumerate(train_batches), total=args.steps, ncols=0):
        key, split_key = random.split(key)
        n_elbo = train_batch(split_key, x_batch, y_batch)

        if i % args.print_interval == 0:
            ws, bs, ls = model.kernel_params.get_params()

            if args.method == "svtp":
                ia, ib = (model.prior.a.safe_value, model.prior.b.safe_value)
                print_str = f"nELBO: {n_elbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}  a: {ia:.4f}  b: {ib:.4f}"
            else:
                print_str = f"nELBO: {n_elbo:.5f}  ws: {ws:.4f}  bs: {bs:.3E}  ls: {ls:.4f}"

            tqdm.write(f"[{i:5d}] " + print_str)

        if i % args.test_interval == 0 or i == args.steps - 1:
            params = model.kernel_params.get_params()
            model.kernel.cache_kernel_fn(params)

            test_nll_list = []
            total_corrects = 0

            for x_batch, y_batch in tqdm(test_batches, leave=False, ncols=0):
                key, split_key = random.split(key)
                nll, corrects = test_batch(split_key, x_batch, y_batch)
                test_nll_list.append(nll)
                total_corrects += corrects

            test_nll = (jnp.sum(jnp.array(test_nll_list)) / num_test).item()
            test_acc = (total_corrects / num_test).item()

            tqdm.write(f"[{i:5d}] NLL: {test_nll:.5f}  ACC: {test_acc:.4f}")

            if test_acc > best_model_acc or (jnp.allclose(test_acc, best_model_acc) and test_nll > best_model_nll):
                update_best = True
            else:
                update_best = False

            if update_best:
                best_step, best_model_acc, best_model_nll = i, test_acc, test_nll
                best_print_str = print_str
                tqdm.write(f"[{i:5d}] Updated: NLL: {test_nll:.5f}  ACC: {test_acc:.4f}")

    print()
    print(f"[{best_step:5d}] NLL: {best_model_nll:.5f}  ACC: {best_model_acc:.4f} " + best_print_str)
    print()
