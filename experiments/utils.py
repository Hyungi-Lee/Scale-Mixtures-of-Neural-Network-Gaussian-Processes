from jax import random
from jax import numpy as jnp


__all__ = [
    "TrainBatch",
    "TestBatch",
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
