"""
Data taken for the UCI Uncertainty Benchmark
repo: https://github.com/yaringal/DropoutUncertaintyExps.git
"""

import os
import jax
import numpy as np
import jax.numpy as jnp


def load_full_data(path):
    data_path = os.path.join(path, "data.txt")
    data = np.loadtxt(data_path)
    return data


def load_train_test_ixs(path, ix):
    ix_train_path = os.path.join(path, f"index_train_{ix}.txt")
    ix_test_path = os.path.join(path, f"index_test_{ix}.txt")
    ix_train = np.loadtxt(ix_train_path, dtype=int)
    ix_test = np.loadtxt(ix_test_path, dtype=int)

    return ix_train, ix_test


def load_features_target_ixs(path):
    features_path = os.path.join(path, "index_features.txt")
    target_path = os.path.join(path, "index_target.txt")

    features_ixs = np.loadtxt(features_path, dtype=int)
    target_ixs = np.loadtxt(target_path, dtype=int)

    return features_ixs, target_ixs


def normalise_features(X_train, X_test):
    xmean, xstd = np.nanmean(X_train, axis=0, keepdims=True), np.nanstd(X_train, axis=0, keepdims=True)
    X_train = (X_train - xmean) / xstd
    X_test = (X_test - xmean) / xstd

    return X_train, X_test, (xmean, xstd)


def normalise_targets(y_train, y_test):
    ymean, ystd = np.nanmean(y_train), np.nanstd(y_train)
    y_train = (y_train - ymean) / ystd
    y_test = (y_test - ymean) / ystd

    return y_train, y_test, (ymean, ystd)


def load_data(path, index):
    data = load_full_data(path)
    train_ixs, test_ixs = load_train_test_ixs(path, index)
    features_ixs, target_ixs = load_features_target_ixs(path)

    X_train = data[np.ix_(train_ixs, features_ixs)]
    y_train = data[np.ix_(train_ixs, target_ixs[None])]

    X_test = data[np.ix_(test_ixs, features_ixs)]
    y_test = data[np.ix_(test_ixs, target_ixs[None])]

    # Normalise dataset
    X_train, X_test, (xmean, xstd) = normalise_features(X_train, X_test)
    y_train, y_test, (ymean, ystd) = normalise_targets(y_train, y_test)

    train = (X_train, y_train.ravel())
    test = (X_test, y_test.ravel())

    dataset = train, test
    dataset = jax.tree_map(jnp.array, dataset)
    dataset = jax.tree_map(jnp.nan_to_num, dataset)
    train, test = dataset

    dataset = {
        "train": train,
        "test": test,
    }
    
    res = {
        "dataset": dataset,
        "ymean": ymean,
        "ystd": ystd,
    }

    return res


if __name__ == "__main__":
    path = (
        "/home/gerardoduran/documents/external"
        "/DropoutUncertaintyExps/UCI_Datasets"
        "/kin8nm"
    )
    path = os.path.join(path, "data")
    data = load_data(path, 0)
    train, test = data
    print(train[0].shape)
