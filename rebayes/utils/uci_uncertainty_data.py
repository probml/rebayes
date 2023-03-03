"""
Data taken for the UCI Uncertainty Benchmark
repo: https://github.com/yaringal/DropoutUncertaintyExps.git
"""

import os
import jax
import numpy as np
import jax.numpy as jnp


def load_raw_data(path):
    data_path = os.path.join(path, "data.txt")
    data = np.loadtxt(data_path)
    data = jax.tree_map(jnp.array, data)
    data = jax.tree_map(jnp.nan_to_num, data)
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


def normalise_features(X, ix_train, ix_test):
    X_train = X[ix_train]
    X_test = X[ix_test]

    xmean, xstd = np.nanmean(X_train, axis=0, keepdims=True), np.nanstd(X_train, axis=0, keepdims=True)
    X_train = (X_train - xmean) / xstd
    X_test = (X_test - xmean) / xstd

    return X_train, X_test, (xmean, xstd)


def normalise_targets(y, ix_train, ix_test):
    y_train = y[ix_train]
    y_test = y[ix_test]

    ymean, ystd = np.nanmean(y_train), np.nanstd(y_train)
    y_train = (y_train - ymean) / ystd
    y_test = (y_test - ymean) / ystd

    return y_train, y_test, (ymean, ystd)


def load_full_data(path):
    data = load_raw_data(path)
    features_ixs, target_ixs = load_features_target_ixs(path)
    X = data[:, features_ixs]
    y = data[:, target_ixs]
    return X, y


def load_folds_data(path, n_partitions=20):
    """
    Load data from all available folds
    """
    X, y = load_full_data(path)

    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []
    coefs_all = []

    for ix in range(n_partitions):
        ix_train, ix_test = load_train_test_ixs(path, ix)
        X_train, X_test, _ = normalise_features(X, ix_train, ix_test)
        y_train, y_test, (ymean, ystd) = normalise_features(y, ix_train, ix_test)
        
        X_test_all.append(X_test)
        y_test_all.append(y_test)
        
        X_train_all.append(X_train)
        y_train_all.append(y_train)
        coefs = {"ymean": ymean.item(), "ystd": ystd.item()}
        coefs_all.append(coefs)
        
    X_train_all = jnp.stack(X_train_all, axis=0)
    y_train_all = jnp.stack(y_train_all, axis=0)

    X_test_all = jnp.stack(X_test_all, axis=0)
    y_test_all = jnp.stack(y_test_all, axis=0)

    train_all = (X_train_all, y_train_all)
    test_all = (X_test_all, y_test_all)

    struct_out = jax.tree_util.tree_structure([0 for e in coefs_all])
    struct_in = jax.tree_util.tree_structure(coefs_all[0])
    coefs_all = jax.tree_util.tree_transpose(struct_out, struct_in, coefs_all)
    coefs_all = jax.tree_map(jnp.array, coefs_all, is_leaf=lambda x: type(x)== list)

    return train_all, test_all, coefs_all


def load_data(path, index):
    data = load_raw_data(path)
    train_ixs, test_ixs = load_train_test_ixs(path, index)
    features_ixs, target_ixs = load_features_target_ixs(path)

    X, y = data[:, features_ixs], data[:, target_ixs]

    X_train, X_test, _ = normalise_features(X, train_ixs, test_ixs)
    y_train, y_test, (ymean, ystd) = normalise_targets(y, train_ixs, test_ixs)

    train = (X_train, y_train.ravel())
    test = (X_test, y_test.ravel())

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
