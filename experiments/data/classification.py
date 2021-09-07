import numpy as np
import tensorflow_datasets as tfds


__all__ = [
    "datasets",
    "get_dataset",
]


image_datasets = [
    "mnist",
    "mnist_corrupted/shot_noise",
    "mnist_corrupted/impulse_noise",
    "mnist_corrupted/spatter",
    "mnist_corrupted/glass_blur",
    "mnist_corrupted/zigzag",
    "fashion_mnist",
    "emnist",
    "kmnist",
    "cifar10",
    "cifar100",
    "svhn_cropped",
]

feature_datasets = [
    "iris",
]

datasets = image_datasets + feature_datasets


def permute_dataset(x, y, seed=0):
    idx = np.random.RandomState(seed).permutation(x.shape[0])
    permuted_x = x[idx]
    permuted_y = y[idx]
    return permuted_x, permuted_y


def _one_hot(x, k):
  return np.array(x[:, None] == np.arange(k), np.float32)


def get_dataset(
    name,
    root="./data",
    num_train=None,
    num_test=None,
    normalize=True,
    dataset_stat=None,
    one_hot=True,
    seed=0,
):
    meta = {}

    if name in feature_datasets:
        ds_builder = tfds.builder(name)
        ds_train, = tfds.as_numpy(
            tfds.load(name, data_dir=root, split=["train"],
                      batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
        )
        x, y = ds_train["features"], ds_train["label"]
        x, y = permute_dataset(x, y, seed=109)

        if num_train is None or num_test is None:
            raise ValueError("Must provide num_train and num_test")

        x_train, y_train = x[:-num_test], y[:-num_test]
        x_test, y_test = x[-num_test:], y[-num_test:]
        meta["type"] = "feature"

    elif name in image_datasets:
        ds_builder = tfds.builder(name)
        ds_train, ds_test = tfds.as_numpy(
            tfds.load(name, data_dir=root, split=["train", "test"],
                      batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
        )
        x_train, y_train = ds_train["image"], ds_train["label"]
        x_test, y_test = ds_test["image"], ds_test["label"]
        meta["type"] = "image"

    else:
        raise KeyError("Unsupported dataset '{}'".format(name))

    num_classes = ds_builder.info.features["label"].num_classes
    meta["num_classes"] = num_classes

    if one_hot:
        y_train = _one_hot(y_train, num_classes)
        y_test = _one_hot(y_test, num_classes)

    x_train, y_train = permute_dataset(x_train, y_train, seed=seed)
    if num_train:
        x_train, y_train = x_train[:num_train], y_train[:num_train]
    if num_test:
        x_test, y_test = x_test[:num_test], y_test[:num_test]

    if normalize:
        if dataset_stat:
            x_mean = dataset_stat["x_mean"]
            x_std  = dataset_stat["x_std"]
        else:
            edim = list(range(x_train.ndim - 1))
            x_mean = np.expand_dims(np.mean(x_train.reshape(-1, x_train.shape[-1]), axis=0), axis=edim)
            x_std  = np.expand_dims(np.std(x_train.reshape(-1, x_train.shape[-1]),  axis=0), axis=edim)
        meta["stat"] = dict(x_mean=x_mean, x_std=x_std)

        x_train = (x_train - x_mean) / x_std
        x_test  = (x_test  - x_mean) / x_std

    else:
        x_train = np.array(x_train).astype(float)
        x_test = np.array(x_test).astype(float)

    return x_train, y_train, x_test, y_test, meta
