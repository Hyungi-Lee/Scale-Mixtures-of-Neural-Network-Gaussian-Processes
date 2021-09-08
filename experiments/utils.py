from typing import Optional
import os
import glob

from tqdm import tqdm

from jax import random
from jax import numpy as jnp

from objax.io import Checkpoint
from objax.variable import VarCollection


__all__ = [
    "TrainBatch",
    "TestBatch",
    "Checkpointer",
    "Logger",
]



class TrainBatch:
    def __init__(self, x, y, batch_size, steps=100, seed=0):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed

    def __iter__(self):
        self.key = random.PRNGKey(self.seed)
        self.step = 0
        return self

    def __len__(self):
        return self.steps

    def __next__(self):
        if self.step >= self.steps:
            raise StopIteration
        else:
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


class Checkpointer(Checkpoint):
    FILE_MATCH: str = "*.npz"
    FILE_FORMAT: str = "%05d.npz"

    def __init__(self, logdir: str, keep_ckpts: int, makedir: bool = True, verbose: bool = True):
        self.logdir = logdir
        self.keep_ckpts = keep_ckpts
        self.verbose = verbose
        if makedir:
            os.makedirs(logdir, exist_ok=True)

    def restore(self, vc: VarCollection, idx: Optional[int] = None):
        assert isinstance(vc, VarCollection), f"Must pass a VarCollection to restore; received type {type(vc)}."
        if idx is None:
            all_ckpts = glob.glob(os.path.join(self.logdir, self.FILE_MATCH))
            if not all_ckpts:
                if self.verbose:
                    print("No checkpoints found. Skipping restoring variables.")
                return 0, ""
            idx = self.checkpoint_idx(max(all_ckpts))
        ckpt = os.path.join(self.logdir, self.FILE_FORMAT % idx)
        if self.verbose:
            print("Resuming from", ckpt)
        self.LOAD_FN(ckpt, vc)
        return idx, ckpt

    def save(self, vc: VarCollection, idx: int):
        assert isinstance(vc, VarCollection), f"Must pass a VarCollection to save; received type {type(vc)}."
        self.SAVE_FN(os.path.join(self.logdir, self.FILE_FORMAT % idx), vc)
        for ckpt in sorted(glob.glob(os.path.join(self.logdir, self.FILE_MATCH)))[:-self.keep_ckpts]:
            os.remove(ckpt)


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
