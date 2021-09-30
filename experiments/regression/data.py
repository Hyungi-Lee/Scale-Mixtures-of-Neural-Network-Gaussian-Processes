import os
import math
import urllib
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.datasets import load_boston


__all__ = [
    "DATASETS",
    "get_dataset",
    "permute_dataset",
    "split_dataset",
]


DATASETS = [
    "boston", "concrete", "energy", "kin8nm", "naval", "plant",
    "wine-red", "wine-white", "yacht", "airfoil", "sic97",
    "syn-normal", "syn-t",
]

DATASET_URLS = {
    "concrete": {
        "Concrete_Data.xls": "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "Concrete_Readme.txt": "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Readme.txt",
    },
    "energy": {
        "ENB2012_data.xlsx": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    },
    "kin8nm": {
        "dataset_2175_kin8nm.csv": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
    },
    "naval": {
        "UCI CBM Dataset.zip": "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
    },
    "plant": {
        "CCPP.zip": "http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
    },
    "wine": {
        "winequality-red.csv": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "winequality-white.csv": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        "winequality.names": "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality.names",
    },
    "yacht": {
        "yacht_hydrodynamics.data": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
    },
    "airfoil": {
        "airfoil_self_noise.dat": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    },
    "parkinsons": {
        "parkinsons_updrs.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
        "parkinsons_updrs.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names",
    },
    "forest": {
        "forestfires.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv",
        "forestfires.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.names",
    },
    "fish": {
        "qsar_fish_toxicity.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv",
    },
    "sic97": {
        "sic97data_01.zip": "https://wiki.52north.org/pub/AI_GEOSTATS/AI_GEOSTATSData/sic97data_01.zip",
    },
}


def _urlretrieve(url, filename, chunk_size = 1024):
    with open(filename, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def _download_url(url, filepath):
    try:
        print("Download {} to {}".format(url, filepath))
        _urlretrieve(url, filepath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead."
                  " Downloading " + url + " to " + filepath)
            _urlretrieve(url, filepath)
        else:
            raise e


def _extract_zip(filepath):
    to_path = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath, "r") as z:
        z.extractall(to_path)


def _download_dataset(name, root):
    root = os.path.expanduser(root)
    dataset_path = os.path.join(root, name)
    os.makedirs(dataset_path, exist_ok=True)

    files = DATASET_URLS[name]

    for filename, url in files.items():
        filepath = os.path.join(dataset_path, filename)

        if not os.path.isfile(filepath):
            _download_url(url, filepath)

            if filename.endswith(".zip"):
                _extract_zip(filepath)


def get_dataset(name, root="./data"):
    if name == "boston":  # Boston Housing
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html
        x, y = load_boston(return_X_y=True)

    elif name == "concrete":  # Concrete Compressive Strength
        # http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
        _download_dataset(name, root)

        filepath = os.path.join(root, "concrete/Concrete_Data.xls")
        excel_data = pd.read_excel(filepath)
        data = excel_data.to_numpy()

        x, y = data[:, :8], data[:, 8]

    elif name == "energy":  # Energy efficiency
        # http://archive.ics.uci.edu/ml/datasets/Energy+efficiency
        _download_dataset(name, root)

        filepath = os.path.join(root, "energy/ENB2012_data.xlsx")
        excel_data = pd.read_excel(filepath)
        data = excel_data.to_numpy()

        x, y = data[:, :8], data[:, 8]

    elif name == "kin8nm":  # kin8nm
        # https://www.openml.org/d/189
        _download_dataset(name, root)

        filepath = os.path.join(root, "kin8nm/dataset_2175_kin8nm.csv")
        csv_data = pd.read_csv(filepath)
        data = csv_data.to_numpy()

        x, y = data[:, :8], data[:, 8]

    elif name == "naval":  # Condition Based Maintenance of Naval Propulsion Plants
        # http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
        _download_dataset(name, root)

        filepath = os.path.join(root, "naval/UCI CBM Dataset/data.txt")
        txt_data = pd.read_table(filepath, delim_whitespace=" ")
        data = txt_data.to_numpy()

        x, y = data[:, :16], data[:, 16]

    elif name == "plant":  # Combined Cycle Power Plant
        # http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
        _download_dataset(name, root)

        filepath = os.path.join(root, "plant/CCPP/Folds5x2_pp.xlsx")
        excel_data = pd.read_excel(filepath)
        data = excel_data.to_numpy()

        x, y = data[:, :4], data[:, 4]

    elif name == "wine-red" or name == "wine-white":  # Wine Quality
        # http://archive.ics.uci.edu/ml/datasets/Wine+Quality
        _download_dataset("wine", root)

        if name == "wine-red":
            filepath = os.path.join(root, "wine/winequality-red.csv")
        else:
            filepath = os.path.join(root, "wine/winequality-white.csv")

        csv_data = pd.read_csv(filepath, delimiter=";")
        data = csv_data.to_numpy()

        x, y = data[:, :11], data[:, 11]

    elif name == "yacht":  # Yacht Hydrodynamics
        # http://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics
        _download_dataset(name, root)

        filepath = os.path.join(root, "yacht/yacht_hydrodynamics.data")
        txt_data = pd.read_table(filepath, delim_whitespace=" ")
        data = txt_data.to_numpy()

        x, y = data[:, :6], data[:, 6]

    elif name == "airfoil":  # Airfoil Self-Noise
        # https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
        _download_dataset(name, root)

        filepath = os.path.join(root, "airfoil/airfoil_self_noise.dat")
        txt_data = pd.read_table(filepath, delim_whitespace="\t", header=None)
        data = txt_data.to_numpy()

        x, y = data[:, :5], data[:, 5]

    elif name == "sic97":  # Switzerland Rainfall
        # https://wiki.52north.org/AI_GEOSTATS/AI_GEOSTATSData
        _download_dataset(name, root)

        filepath = os.path.join(root, "sic97/sic_full.dat")
        txt_data = pd.read_table(filepath, sep=",", index_col=0, skiprows=6, header=None)
        data = txt_data.to_numpy()

        x, y = data[:, :2], data[:, 2]

    elif name == "syn-normal":
        num = 100
        rs = np.random.RandomState(829)

        x = np.linspace(-num / 2, num / 2, num)[:, None]
        cov = np.exp(-0.5 * (x - x.T) ** 2)

        y = rs.multivariate_normal(mean=np.zeros(num), cov=cov, size=1).flatten() \
          + rs.standard_normal(size=num) * 0.2

    elif name == "syn-t":
        num = 300
        rs = np.random.RandomState(761)

        x = np.linspace(-num / 2, num / 2, num)[:, None]
        cov = np.exp(-0.5 * (x - x.T) ** 2)
        y = rs.multivariate_normal(mean=np.zeros(num), cov=cov, size=1).flatten() \
          + rs.standard_t(df=1, size=num) * 0.8

    else:
        raise KeyError("Unsupported dataset '{}'".format(name))

    return x, y


def split_dataset(x, y, train, valid, test, normalize_x=True, normalize_y=True):
    fractions = train + valid + test

    if not math.isclose(fractions, 1.0) and fractions > 1.0:
        raise ValueError("Sum of fractions exceed 1.0")

    train_num = int(train * len(x))
    x_train = x[:train_num]
    y_train = y[:train_num]

    valid_num = int(valid * len(x))
    x_valid = x[train_num: train_num + valid_num]
    y_valid = y[train_num: train_num + valid_num]

    if math.isclose(fractions, 1.0):
        x_test = x[train_num + valid_num:]
        y_test = y[train_num + valid_num:]
    elif fractions < 1.0:
        test_num = int(test * len(x))
        x_test = x[train_num + valid_num: train_num + valid_num + test_num]
        y_test = y[train_num + valid_num: train_num + valid_num + test_num]

    if normalize_x:
        x_std = np.std(x_train, axis=0)
        x_mean = np.mean(x_train, axis=0)

        x_train = (x_train - x_mean) / x_std
        x_valid = (x_valid - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std

        np.nan_to_num(x_train, copy=False)
        np.nan_to_num(x_valid, copy=False)
        np.nan_to_num(x_test, copy=False)

    if normalize_y:
        y_std = np.std(y_train, axis=0)
        y_mean = np.mean(y_train, axis=0)

        y_train = (y_train - y_mean) / y_std
        y_valid = (y_valid - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
    else:
        y_std = 1.
        y_mean = 0.

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test), (y_std, y_mean)


def permute_dataset(x, y, seed=0):
    idx = np.random.RandomState(seed).permutation(x.shape[0])
    permuted_x, permuted_y = x[idx], y[idx]
    return permuted_x, permuted_y
