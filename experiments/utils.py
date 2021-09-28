from typing import Optional

import os
import glob
import math
import random

from tqdm import tqdm

import numpy as np

from objax.variable import VarCollection
from objax.io.ops import save_var_collection


__all__ = [
    "get_context_summary",
    "TrainBatch",
    "TestBatch",
    "Checkpointer",
    "Logger",
    "ReduceLROnPlateau",
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


class DataLoader:
    def __init__(self, x, y, batch_size: Optional[int] = None, *, shuffle: bool = False, seed: int = 0):
        self.shuffle = shuffle
        self.seed = seed

        self.x = np.array(x)
        self.y = np.array(y)
        self.indices = list(range(x.shape[0]))
        self.batch_size = (x.shape[0] if batch_size is None else batch_size)

        self.not_use_indices = (batch_size is None and not shuffle)
        self._batch_indices = None
        self._batch_idx = None

    def __iter__(self):
        if self.shuffle:
            self.seed += 1
            indices = self.indices.copy()
            random.Random(self.seed).shuffle(indices)
        else:
            indices = self.indices

        self._batch_idx = 0
        if not self.not_use_indices:
            self._batch_indices = [indices[i: i + self.batch_size]
                                   for i in range(0, len(indices), self.batch_size)]
        return self

    def __next__(self):
        if self.not_use_indices:
            if self._batch_idx > 0:
                raise StopIteration
        else:
            if self._batch_idx >= len(self._batch_indices):
                raise StopIteration

        if self.not_use_indices:
            x_batch = self.x
            y_batch = self.y
        else:
            indices = self._batch_indices[self._batch_idx]
            x_batch = self.x[indices]
            y_batch = self.y[indices]

        self._batch_idx += 1

        return x_batch, y_batch

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    @property
    def num_data(self):
        return self.x.shape[0]


class Checkpointer:
    FILE_MATCH: str = "*.npz"
    FILE_FORMAT: str = "{:03d}.npz"

    def __init__(
        self,
        logdir: str,
        keep_ckpts: int = 10,
        makedir: bool = True,
    ):
        self.logdir = logdir
        self.keep_ckpts = keep_ckpts
        if makedir:
            os.makedirs(logdir, exist_ok=True)
        self.best_loss = float("inf")

    def save(self, idx: int, vc: VarCollection):
        assert isinstance(vc, VarCollection), f"Must pass a VarCollection to save; received type {type(vc)}."
        save_var_collection(os.path.join(self.logdir, self.FILE_FORMAT.format(idx)), vc)
        for ckpt in sorted(glob.glob(os.path.join(self.logdir, self.FILE_MATCH)))[:-self.keep_ckpts]:
            os.remove(ckpt)

    def step(self, idx, loss: float, vc):
        if loss < self.best_loss:
            self.best_loss = loss
            self.save(idx, vc)
            updated = True
        else:
            updated = False
        return updated


class Logger:
    def __init__(self, logdir: str, filename: str = "train.log", makedir: bool = True, quite: bool = False):
        self.logdir = logdir
        self.quite = quite
        if makedir:
            os.makedirs(logdir, exist_ok=True)
        self.filename = filename
        self.logfile = open(os.path.join(logdir, self.filename), "w")

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


class ReduceLROnPlateau:
    def __init__(self, lr, mode="min", factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode="rel",
                 min_lr=0, eps=1e-8, verbose=False):

        self.lr = lr
        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics):
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        reduced = False

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            reduced = True

        return reduced

    def _reduce_lr(self):
        old_lr = self.lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.lr = new_lr

    def is_better(self, a, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == "max" and epsilon_mode == "abs":
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = float("inf")
        else:  # mode == "max":
            self.mode_worse = -float("inf")

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
