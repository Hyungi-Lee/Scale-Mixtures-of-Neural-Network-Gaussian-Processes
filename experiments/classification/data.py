import numpy as np
import tensorflow_datasets as tfds


__all__ = [
    "datasets",
    "get_train_dataset",
    "get_test_dataset",
]


image_datasets = [
    "mnist",
    "mnist_imbalanced",
    "fashion_mnist",
    "kmnist",
    "cifar10",
    "cifar100",
    "svhn_cropped",
    "emnist",
    "emnist/letters",
    "mnist_corrupted/shot_noise",
    "mnist_corrupted/impulse_noise",
    "mnist_corrupted/spatter",
    "mnist_corrupted/glass_blur",
    "mnist_corrupted/zigzag",
    "cifar10_corrupted/spatter_5",
    "cifar10_corrupted/brightness_5",
    "cifar10_corrupted/fog_5",
    "cifar10_corrupted/defocus_blur_5",
    "cifar10_corrupted/frosted_glass_blur_5",
    "cifar10_corrupted/gaussian_noise_5",
    "cifar10_imbalanced",
]

feature_datasets = [
    "iris",
]

datasets = image_datasets + feature_datasets


def permute_dataset(x, y, seed=0):
    idx = np.random.RandomState(seed).permutation(x.shape[0])
    permuted_x, permuted_y = x[idx], y[idx]
    return permuted_x, permuted_y


def get_num_class_data(num_data_per_class, num_class, mode="exp", factor=1):
    if mode == "exp":
        d = np.exp(np.arange(num_class) * factor)
    elif mode == "step":
        d = np.arange(0, num_class) + 1 / factor
    else:
        raise ValueError("Unknown mode")

    d = d / np.max(d) * num_data_per_class
    d = np.round(d).astype(int).tolist()
    return d


def get_train_dataset(
    name,
    root="./data",
    num_data=None,
    valid_prop=0.1,
    normalize=True,
    seed=0,
):
    meta = {}

    if name.endswith("_imbalanced"):
        imbalanced = True
        name = name[:-11]
    else:
        imbalanced = False

    if name in feature_datasets:
        ds_builder = tfds.builder(name)
        ds_train, = tfds.as_numpy(
            tfds.load(name, data_dir=root, split=["train"],
                      batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
        )
        x, y = ds_train["features"], ds_train["label"]
        x, y = permute_dataset(x, y, seed=109)

        if num_data is None:
            raise ValueError("Must provide num_train and num_test")

        x_train, y_train = x[:num_data], y[:num_data]
        meta["type"] = "feature"

    elif name in image_datasets:
        ds_builder = tfds.builder(name)
        ds_train, = tfds.as_numpy(
            tfds.load(name, data_dir=root, split=["train"],
                      batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
        )
        x_train, y_train = ds_train["image"], ds_train["label"]
        meta["type"] = "image"

    else:
        raise KeyError("Unsupported dataset '{}'".format(name))

    num_class = ds_builder.info.features["label"].num_classes
    meta["num_class"] = num_class

    x_train, y_train = permute_dataset(x_train, y_train, seed=seed)
    if num_data is None:
        num_data = x_train.shape[0]

    if imbalanced:
        data_class = []
        for class_idx in range(num_class):
            idxs = [y_train == class_idx]
            data_class.append((x_train[idxs], y_train[idxs]))
        min_num_data_per_class = min(map(lambda v: len(v[0]), data_class))
        num_train_per_class = int(min_num_data_per_class * (1 - valid_prop))
        num_valid_per_class = min_num_data_per_class - num_train_per_class
        num_class_data = get_num_class_data(min_num_data_per_class, num_class, mode="exp", factor=.5)
        num_valid = num_class * num_valid_per_class
        num_train = sum(num_class_data)
        assert num_valid > 0, "num_valid must be > 0"

        x_valid = np.concatenate([x[-num_valid_per_class:] for (x, _) in data_class])
        y_valid = np.concatenate([y[-num_valid_per_class:] for (_, y) in data_class])
        x_train = np.concatenate([x[:num_train] for num_train, (x, _) in zip(num_class_data, data_class)])
        y_train = np.concatenate([y[:num_train] for num_train, (_, y) in zip(num_class_data, data_class)])
        x_train, y_train = permute_dataset(x_train, y_train, seed=seed)

    else:
        num_valid = int(num_data * valid_prop)
        num_train = num_data - num_valid
        assert num_valid > 0, "num_valid must be > 0"

        x_train, y_train = x_train[:num_train], y_train[:num_train]
        x_valid, y_valid = x_train[-num_valid:], y_train[-num_valid:]

    if normalize:
        edim = list(range(x_train.ndim - 1))
        if "cifar" in name:
            x_mean = np.expand_dims(np.array((0.4914, 0.4822, 0.4465)) * 255., axis=edim)
            x_std  = np.expand_dims(np.array((0.2023, 0.1994, 0.2010)) * 255., axis=edim)
        else:
            x_data = np.concatenate((x_train, x_valid), axis=0)
            x_mean = np.expand_dims(np.mean(x_data.reshape(-1, x_data.shape[-1]), axis=0), axis=edim)
            x_std  = np.expand_dims(np.std(x_data.reshape(-1, x_data.shape[-1]),  axis=0), axis=edim)
        x_train = (x_train - x_mean) / x_std
        x_valid = (x_valid - x_mean) / x_std
        meta["stat"] = dict(x_mean=x_mean, x_std=x_std)
    else:
        x_train = np.array(x_train).astype(float)
        x_valid = np.array(x_valid).astype(float)

    return x_train, y_train, x_valid, y_valid, meta


def get_test_dataset(
    name,
    root="./data",
    num_test=None,
    normalize=True,
    dataset_stat=None,
):
    meta = {}

    if name.endswith("_imbalanced"):
        raise KeyError("Test dataset doesn't support imbalanced dataset")

    if name in feature_datasets:
        ds_builder = tfds.builder(name)
        ds_train, = tfds.as_numpy(
            tfds.load(name, data_dir=root, split=["train"],
                      batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
        )
        x, y = ds_train["features"], ds_train["label"]
        x, y = permute_dataset(x, y, seed=109)

        if num_test is None or num_test is None:
            raise ValueError("Must provide num_train and num_test")

        x_test, y_test = x[-num_test:], y[-num_test:]
        meta["type"] = "feature"

    elif name in image_datasets:
        ds_builder = tfds.builder(name)
        ds_test, = tfds.as_numpy(
            tfds.load(name, data_dir=root, split=["test"],
                      batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
        )
        x_test, y_test = ds_test["image"], ds_test["label"]
        meta["type"] = "image"

    else:
        raise KeyError("Unsupported dataset '{}'".format(name))

    num_class = ds_builder.info.features["label"].num_classes
    meta["num_class"] = num_class

    if num_test:
        x_test, y_test = x_test[:num_test], y_test[:num_test]

    if normalize:
        x_mean = dataset_stat["x_mean"]
        x_std  = dataset_stat["x_std"]
        x_test  = (x_test  - x_mean) / x_std
    else:
        x_test = np.array(x_test).astype(float)

    return x_test, y_test, meta
