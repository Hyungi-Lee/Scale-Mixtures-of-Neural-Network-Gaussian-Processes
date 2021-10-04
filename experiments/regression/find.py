import numpy as np
import pandas as pd

from jax import jit
from jax import numpy as jnp
from jax.scipy import stats
from jax.nn import logsumexp

from scipy import stats as scipy_stats

from neural_tangents.predict import gradient_descent_mse_ensemble

from .data import DATASETS, get_dataset, permute_dataset, split_dataset
from ..nt_kernels import get_mlp_kernel, get_dense_resnet_kernel
from ..utils import Logger, get_context_summary


WSL = [1, 1.4, 2]
BSL = [0, 0.3, 1]
EL = [float(f"1e{v}") for v in range(-6, 5)]
AL = [1, 2, 3]
BL = [1, 2, 3]


def add_subparser(subparsers):
    parser = subparsers.add_parser("find", aliases=["fd"])
    parser.set_defaults(func=main)

    parser.add_argument("-n",   "--network",          choices=["resnet", "mlp"], default=None)
    parser.add_argument("-dn",  "--data-name",        choices=DATASETS, required=True)
    parser.add_argument("-dr",  "--data-root",        type=str, default="./data")
    parser.add_argument("-cd",  "--ckpt-dir",         type=str, required=True, default="_test/reg-find/")

    parser.add_argument("-vp",  "--valid-prop",       type=float, default=0.1)
    parser.add_argument("-nd",  "--num-data",         type=int, default=None)

    parser.add_argument("-al",  "--alpha-list",       type=float, nargs="+", default=AL)
    parser.add_argument("-bl",  "--beta-list",        type=float, nargs="+", default=BL)
    parser.add_argument("-el",  "--eps-list",         type=float, nargs="+", default=EL)

    parser.add_argument("-nh",  "--num-hiddens",      type=int, default=4)
    parser.add_argument("-act", "--activation",       choices=["erf", "relu"], default="relu")
    parser.add_argument("-wsl", "--w-std-list",      type=float, nargs="+", default=WSL)
    parser.add_argument("-bsl", "--b-std-list",      type=float, nargs="+", default=BSL)

    parser.add_argument("-s",   "--seed",             type=int, default=10)
    parser.add_argument("-q",   "--quite",            default=False, action="store_true")
    parser.add_argument("-c",   "--comment",          type=str, default="")


@jit
def gaussian_nll(y, mean, cov):
    sigma = jnp.sqrt(jnp.diag(cov))
    logpdf = stats.norm.logpdf(y, mean, sigma)
    nll = -jnp.mean(logpdf)
    return nll


@jit
def matmul3(a, b, c):
    return jnp.einsum("i,ij,j->", a, b, c)


def get_K(base_kernel_fn, num_hiddens, act, x_train):
    def get_kernel_fn(w_std, b_std):
        return base_kernel_fn(num_hiddens, act=act, w_std=w_std, b_std=b_std, last_w_std=1.)
    def K(w_std, b_std):
        cov_data = get_kernel_fn(w_std, b_std)(x_train, x_train, get="nngp")
        return cov_data
    return jit(K)


def get_predict(kernel_fn, x_train, y_train, x_test):
    def predict(eps):
        predict_fn = gradient_descent_mse_ensemble(kernel_fn, x_train, y_train[:,None], diag_reg=eps)
        mean, cov = predict_fn(x_test=x_test, get="nngp", compute_cov=True)
        return mean, cov
    return jit(predict)


def main(args):
    logger = Logger(args.ckpt_dir, quite=args.quite)

    logger.log(get_context_summary(args, {}))

    try:
        # Dataset
        x_d, y_d = get_dataset(name=args.data_name, root=args.data_root)
        x_d, y_d = permute_dataset(x_d, y_d, seed=10)
        splits = split_dataset(x_d, y_d, train=0.8, valid=0.1, test=0.1)
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (y_std, y_mean) = splits

        num_train = x_train.shape[0]

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

        x = x_test
        y = y_test

        y_ = (y * y_std) + y_mean

        min_t, min_tv = None, float("inf")
        min_g, min_gv = None, float("inf")

        K = get_K(base_kernel_fn, args.num_hiddens, args.activation, x_train)

        il = len(args.w_std_list)
        jl = len(args.b_std_list)
        kl = len(args.eps_list)
        total = il * jl * kl

        minus_log_two_pi = -(num_train / 2) * jnp.log(2 * jnp.pi)

        for i, w_std in enumerate(args.w_std_list):
            for j, b_std in enumerate(args.b_std_list):
                kernel_fn = base_kernel_fn(args.num_hiddens, act=args.activation, w_std=w_std, b_std=b_std, last_w_std=1.)
                predict = get_predict(kernel_fn, x_train, y_train, x)
                cov_data = K(w_std, b_std)

                for k, eps in enumerate(args.eps_list):
                    mean, cov = predict(eps)


                    mean_ = (mean.flatten() * y_std) + y_mean
                    cov_ = cov * y_std ** 2

                    gnll = gaussian_nll(y_, mean_, cov_)
                    if gnll < min_gv:
                        min_g, min_gv = (w_std, b_std, eps), gnll

                    cov_data_eps = cov_data + np.eye(num_train) * eps
                    inv_cov_data_std = jnp.linalg.inv(cov_data_eps)
                    d_std = matmul3(y_train, inv_cov_data_std, y_train)
                    minus_y_train_K_NNGP_y_train = -(1/2) * d_std

                    try:
                        error = False
                        minus_log_det_K_NNGP = scipy_stats.multivariate_normal.logpdf(y_train, None, cov_data_eps, allow_singular=True) \
                                             - minus_log_two_pi - minus_y_train_K_NNGP_y_train
                    except:
                        error = True

                    if error:
                        continue

                    std_diag = jnp.sqrt(jnp.diag(cov))

                    table = []
                    for a in args.alpha_list:
                        col = []
                        for b in args.beta_list:
                            sample_q = scipy_stats.burr12.rvs(c=a, d=b, loc=0., scale=1., size=1000, random_state=101)
                            minus_log_sigma = -(1/2)*num_train*jnp.log(sample_q)

                            prob_prior = scipy_stats.burr12.pdf(sample_q, c=a, d=b, loc=0., scale=1.)
                            prob_q = scipy_stats.burr12.pdf(sample_q, c=a, d=b, loc=0., scale=1.)

                            log_prob_data = minus_log_two_pi + minus_log_det_K_NNGP + minus_y_train_K_NNGP_y_train / sample_q + minus_log_sigma

                            prob_data = jnp.exp(log_prob_data - log_prob_data.max())
                            prob_joint = prob_data * prob_prior
                            w = prob_joint / prob_q
                            w_bar = w / jnp.sum(w)
                            std = jnp.sqrt(sample_q[:, None]) * std_diag[None, :]  # (S, B)

                            log_probs = jnp.log(w_bar + 1e-24)[:, None] + scipy_stats.norm.logpdf(y_, mean_, std * y_std)  # (S, B)
                            tnll = -jnp.mean(logsumexp(log_probs, axis=0))
                            if tnll < min_tv:
                                min_t, min_tv = (w_std, b_std, a, b, eps), tnll
                            col.append(tnll.item())
                        table.append(col)
                    logger.log(f"\n{w_std}-{b_std}-{eps}: {i * jl * kl + j * kl + k+1} / {total} ({i+1}/{il}, {j+1}/{jl}, {k+1}/{kl})")
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.max_colwidth', -1)
                    logger.log(f"{pd.DataFrame(table, index=args.alpha_list, columns=args.beta_list).round(4)}\n")
                    logger.log(f"({min_t}): {min_tv:.4f}")
                    logger.log(f"({min_g}): {min_gv:.4f}")

        logger.log(f"({min_t}): {min_tv:.4f}")
        logger.log(f"({min_g}): {min_gv:.4f}")

    except KeyboardInterrupt:
        raise KeyboardInterrupt

    except:
        import traceback
        logger.log(f"\n{traceback.format_exc()}\nStopped")

    finally:
        logger.close()
