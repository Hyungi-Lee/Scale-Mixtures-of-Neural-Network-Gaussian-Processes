import os
import glob
import math

from tqdm import tqdm

from jax import random
from jax import numpy as jnp

from objax.variable import VarCollection
from objax.io.ops import load_var_collection, save_var_collection


__all__ = [
    "get_context_summary",
    "TrainBatch",
    "TestBatch",
    "Checkpointer",
    "Logger",
]


def get_context_summary(args, values_dict, indent=2):
    args_dict = {k: v for k, v in vars(args).items() if k != "func"}
    key_max_len = max(map(len, list(args_dict.keys()) + list(values_dict.keys())))

    s = "Args:\n"
    for k, v in args_dict.items():
        s += f"{' ' * indent}{k.ljust(key_max_len)}: {v}\n"

    s += "\nValues:\n"
    for k, v in values_dict.items():
        s += f"{' ' * indent}{k.ljust(key_max_len)}: {v}\n"

    s += "\n"
    return s


class TrainBatch:
    def __init__(self, x, y, batch_size, seed=0):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        self.key = random.PRNGKey(self.seed)
        self.step = 0
        return self

    def __next__(self):
        self.step += 1
        self.key, split = random.split(self.key)

        random_idxs = random.permutation(split, jnp.arange(self.x.shape[0], dtype=int))
        random_x = self.x[random_idxs]
        random_y = self.y[random_idxs]

        x_batch = random_x[:self.batch_size]
        y_batch = random_y[:self.batch_size]

        return x_batch, y_batch


class TestBatch:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.batch_len = (x.shape[0] // batch_size) + (1 if x.shape[0] % batch_size else 0)

    def __iter__(self):
        self.batch_i = 0
        return self

    def __len__(self):
        return self.batch_len

    def __next__(self):
        if self.batch_i >= self.batch_len:
            raise StopIteration

        batch_start = self.batch_i * self.batch_size
        batch_end = batch_start + self.batch_size
        x_batch = self.x[batch_start: batch_end]
        y_batch = self.y[batch_start: batch_end]

        self.batch_i += 1

        return x_batch, y_batch


class Checkpointer:
    FILE_MATCH: str = "*.npz"
    FILE_FORMAT: str = "{:06d}.npz"

    def __init__(
        self,
        logdir: str,
        keep_ckpts: int = 10,
        makedir: bool = True,
        patience: int = 10,
    ):
        self.logdir = logdir
        self.keep_ckpts = keep_ckpts
        self.patience = patience
        if makedir:
            os.makedirs(logdir, exist_ok=True)
        self.best_losses = [float("inf")]

    def save(self, idx: int, vc: VarCollection):
        assert isinstance(vc, VarCollection), f"Must pass a VarCollection to save; received type {type(vc)}."
        save_var_collection(os.path.join(self.logdir, self.FILE_FORMAT.format(idx)), vc)
        for ckpt in sorted(glob.glob(os.path.join(self.logdir, self.FILE_MATCH)))[:-self.keep_ckpts]:
            os.remove(ckpt)

    def step(self, idx, loss: float, vc):
        updated, stop = False, False

        if loss < self.best_losses[0]:
            self.save(idx, vc)
            updated = True
        elif loss > self.best_losses[-1] and len(self.best_losses) == self.patience:
            stop = True
        elif math.isnan(loss):
            stop = True

        self.best_losses = sorted(self.best_losses + [loss])[:self.patience]
        return updated, stop


class Logger:
    FILE_NAME: str = "train.log"

    def __init__(self, logdir: str, makedir: bool = True, quite: bool = False):
        self.logdir = logdir
        self.quite = quite
        if makedir:
            os.makedirs(logdir, exist_ok=True)
        self.logfile = open(os.path.join(logdir, self.FILE_NAME), "w")

    def log(self, *args, is_tqdm: bool = False):
        s = "".join(map(str, args))
        self.logfile.write(s + "\n")
        if not self.quite:
            if is_tqdm:
                tqdm.write(s)
            else:
                print(s, flush=True)
        self.logfile.flush()

    def close(self):
        self.logfile.close()
