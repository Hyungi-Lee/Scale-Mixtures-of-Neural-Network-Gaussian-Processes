import numpy as np
import tensorflow_datasets as tfds
from jax.nn import one_hot


__all__ = [
    "DATASETS",
    "get_train_dataset",
    "get_test_dataset",
]


DATASETS_DICT = {
    # MNIST
    "mnist/default": ("mnist", "default", None),
    "mnist/ood": ("mnist", "ood", (1,4,8)),
    "mnist/imbalanced": ("mnist", "imbalanced", ("exp", .5)),
    "mnist/noisy_label": ("mnist", "noisy_label", 0.1),
    "mnist/shot_noise": ("mnist_corrupted/shot_noise", "corrupted", None),
    "mnist/impulse_noise": ("mnist_corrupted/impulse_noise", "corrupted", None),
    "mnist/spatter": ("mnist_corrupted/spatter", "corrupted", None),
    "mnist/glass_blur": ("mnist_corrupted/glass_blur", "corrupted", None),
    "mnist/zigzag": ("mnist_corrupted/zigzag", "corrupted", None),
    # KMNIST
    "kmnist/default": ("kmnist", "default", None),
    "kmnist/ood": ("kmnist", "ood", (1,4,8)),
    "kmnist/imbalanced": ("kmnist", "imbalanced", ("exp", .5)),
    "kmnist/noisy_label": ("kmnist", "noisy_label", 0.1),
    # Fashion MNIST
    "fashion_mnist/default": ("fashion_mnist", "default", None),
    "fashion_mnist/ood": ("fashion_mnist", "ood", (1,4,8)),
    "fashion_mnist/imbalanced": ("fashion_mnist", "imbalanced", ("exp", .5)),
    "fashion_mnist/noisy_label": ("fashion_mnist", "noisy_label", 0.1),
    # EMNIST
    "emnist/default": ("emnist/letters", "default", None),
    "emnist/ood": ("emnist/letters", "ood", (1,4,8)),
    "emnist/imbalanced": ("emnist/letters", "imbalanced", ("exp", .5)),
    "emnist/noisy_label": ("emnist/letters", "noisy_label", 0.1),
    # CIFAR10
    "cifar10/default": ("cifar10", "default", None),
    "cifar10/ood": ("cifar10", "ood", (1,4,8)),
    "cifar10/imbalanced": ("cifar10", "imbalanced", ("exp", .5)),
    "cifar10/noisy_label": ("cifar10", "noisy_label", 0.1),
    "cifar10/fog_1": ("cifar10_corrupted/fog_1", "corrupted", None),
    "cifar10/fog_5": ("cifar10_corrupted/fog_5", "corrupted", None),
    "cifar10/impulse_noise_1": ("cifar10_corrupted/impulse_noise_1", "corrupted", None),
    "cifar10/impulse_noise_5": ("cifar10_corrupted/impulse_noise_5", "corrupted", None),
    "cifar10/shot_noise_1": ("cifar10_corrupted/shot_noise_1", "corrupted", None),
    "cifar10/shot_noise_5": ("cifar10_corrupted/shot_noise_5", "corrupted", None),
    "cifar10/spatter_1": ("cifar10_corrupted/spatter_1", "corrupted", None),
    "cifar10/spatter_5": ("cifar10_corrupted/spatter_5", "corrupted", None),
    "cifar10/frost_1": ("cifar10_corrupted/frost_1", "corrupted", None),
    "cifar10/frost_5": ("cifar10_corrupted/frost_5", "corrupted", None),
    "cifar10/snow_1": ("cifar10_corrupted/snow_1", "corrupted", None),
    "cifar10/snow_5": ("cifar10_corrupted/snow_5", "corrupted", None),
    # SVHN
    "svhn/default": ("svhn_cropped", "default", None),
    "svhn/ood": ("svhn_cropped", "ood", (1,4,8)),
    "svhn/imbalanced": ("svhn_cropped", "imbalanced", ("exp", .5)),
    "svhn/noisy_label": ("svhn_cropped", "noisy_label", 0.1),
}

DATASETS = list(DATASETS_DICT.keys())

DATASET_FORMATTER = {
    "ood": lambda option: ",".join(map(str, option)),
    "imbalanced": lambda option: f"{option[0]}{option[1]}",
    "noisy_label": lambda option: str(option),
}


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


def parse_dataset(name):
    tokens = name.split("/")

    base_name = tokens[0]
    detail_name = tokens[1] if len(tokens) > 1 else "default"
    option = None

    if len(tokens) > 2:
        if detail_name == "ood":
            try:
                option = list(map(int, tokens[2].split(",")))
            except:
                raise ValueError(f"Invalid OOD option: {tokens[2]}")
        elif detail_name == "imbalanced":
            if tokens[2].startswith("exp"):
                option = ("exp", float(tokens[2][3:]))
            elif tokens[2].startswith("step"):
                option = ("step", float(tokens[2][4:]))
            else:
                raise ValueError(f"Invalid imbalanced option {tokens[2]}")
        elif detail_name == "noisy_label":
            try:
                option = float(tokens[2])
            except:
                raise ValueError(f"Invalid noisy label option: {tokens[2]}")

    dname = f"{base_name}/{detail_name}"

    if dname in DATASETS_DICT:
        base, detail, default_option = DATASETS_DICT[dname]
        if option is None:
            option = default_option

        clean_name = dname
        if option is not None:
            clean_name += "/" + DATASET_FORMATTER[detail](option)
    else:
        raise ValueError(f"Unsupported dataset: {dname}")

    return (base, detail, option), clean_name


def normalize_dataset(name, x_data):
    if "mnist" in name:
        x_mean = np.array((0.5,))
        x_std = np.array((0.5,))
    elif "cifar" in name or "svhn" in name:
        x_mean = np.array((0.4914, 0.4822, 0.4465))
        x_std  = np.array((0.2023, 0.1994, 0.2010))

    edim = list(range(x_data.ndim - 1))
    x_mean = np.expand_dims(x_mean, axis=edim)
    x_std = np.expand_dims(x_std, axis=edim)
    x_data = (x_data - x_mean) / x_std

    return x_data


def get_train_dataset(name, root="./data", num_data=None, valid_prop=0.1, normalize=True, onehot=False, seed=0):
    (base, detail, option), clean_name = parse_dataset(name)

    ds_builder = tfds.builder(base)
    ds_train, = tfds.as_numpy(
        tfds.load(base, data_dir=root, split=["train"],
                  batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
    )

    x_data, y_data = ds_train["image"], ds_train["label"]
    num_class = ds_builder.info.features["label"].num_classes
    x_data = x_data / 255.

    debug_msg = ""

    if detail == "noisy_label":
        noise_prob = option
        idx = (np.random.RandomState(seed).uniform(size=y_data.shape[0]) < noise_prob)
        noise_label = np.random.RandomState(seed).randint(num_class, size=np.sum(idx))
        y_data[idx] = noise_label
        debug_msg = f"{np.sum(idx)} / {y_data.shape[0]} (noisy labels)"
    elif detail == "ood":
        out_labels = option
        idx = np.all(np.vstack([(y_data != label)[None, :] for label in out_labels]), axis=0)
        x_data, y_data = x_data[idx], y_data[idx]
        data_in_class = [str(np.sum(y_data == label)) for label in range(num_class)]
        debug_msg = str(data_in_class) + " (data / class)"

    x_data, y_data = permute_dataset(x_data, y_data, seed=seed)

    if num_data is None:
        num_data = x_data.shape[0]

    if detail == "imbalanced":
        mode, factor = option
        data_class = []
        for class_idx in range(num_class):
            idx = [y_data == class_idx]
            data_class.append((x_data[idx], y_data[idx]))
        min_num_data_per_class = min(map(lambda v: v[0].shape[0], data_class))
        num_train_per_class = int(min_num_data_per_class * (1 - valid_prop))
        num_valid_per_class = min_num_data_per_class - num_train_per_class
        num_class_data = get_num_class_data(min_num_data_per_class, num_class, mode=mode, factor=factor)
        num_valid = num_class * num_valid_per_class
        num_train = sum(num_class_data)

        if num_valid:
            x_valid = np.concatenate([x[-num_valid_per_class:] for (x, _) in data_class])
            y_valid = np.concatenate([y[-num_valid_per_class:] for (_, y) in data_class])
        x_train = np.concatenate([x[:num_train] for num_train, (x, _) in zip(num_class_data, data_class)])
        y_train = np.concatenate([y[:num_train] for num_train, (_, y) in zip(num_class_data, data_class)])
        x_train, y_train = permute_dataset(x_train, y_train, seed=seed)

        if num_valid:
            data_in_class = [str(np.sum(y_train == label) + np.sum(y_valid == label)) for label in range(num_class)]
        else:
            data_in_class = [str(np.sum(y_train == label)) for label in range(num_class)]

        debug_msg = str(data_in_class) + " (data / class)"

    else:
        num_valid = int(num_data * valid_prop)
        num_train = num_data - num_valid

        x_train, y_train = x_data[:num_train], y_data[:num_train]
        if num_valid:
            x_valid, y_valid = x_data[-num_valid:], y_data[-num_valid:]

    if normalize:
        x_train = normalize_dataset(clean_name, x_train)
        if num_valid:
            x_valid = normalize_dataset(clean_name, x_valid)

    if onehot:
        y_train = one_hot(y_train, num_class)
        if num_valid:
            y_valid = one_hot(y_valid, num_class)

    if num_valid:
        return (x_train, y_train), (x_valid, y_valid), (num_class, clean_name, debug_msg)
    else:
        return (x_train, y_train), (num_class, clean_name, debug_msg)


def get_test_dataset(name, root="./data", num_data=None, normalize=True, onehot=False):
    (base, detail, _), clean_name = parse_dataset(name)

    if detail in ["ood", "imbalanced", "noisy_label"]:
        raise KeyError(f"Test dataset doesn't support {detail} dataset")

    ds_builder = tfds.builder(base)
    ds_test, = tfds.as_numpy(
        tfds.load(base, data_dir=root, split=["test"],
                  batch_size=-1, as_dataset_kwargs=dict(shuffle_files=False))
    )

    x_test, y_test = ds_test["image"], ds_test["label"]
    num_class = ds_builder.info.features["label"].num_classes
    x_test = x_test / 255.

    if num_data is not None:
        x_test, y_test = permute_dataset(x_test, y_test, seed=109)
        x_test, y_test = x_test[:num_data], y_test[:num_data]

    if normalize:
        x_test = normalize_dataset(clean_name, x_test)

    if onehot:
        y_test = one_hot(y_test, num_class)

    return (x_test, y_test), (num_class, clean_name)
