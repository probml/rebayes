"""
Data taken for the UCI Uncertainty Benchmark
repo: https://github.com/yaringal/DropoutUncertaintyExps.git
"""

import os
import jax
import numpy as np
import jax.numpy as jnp

def load_data(path, index):
    data_path = os.path.join(path, "data.txt")
    ixs_train_path = os.path.join(path, f"index_train_{index}.txt")
    ixs_test_path = os.path.join(path, f"index_test_{index}.txt")
    features_path = os.path.join(path, "index_features.txt")
    target_path = os.path.join(path, "index_target.txt")

    data = np.loadtxt(data_path)
    train_ixs = np.loadtxt(ixs_train_path, dtype=int)
    test_ixs = np.loadtxt(ixs_test_path, dtype=int)
    features_ixs = np.loadtxt(features_path, dtype=int)
    target_ixs = np.loadtxt(target_path, dtype=int)

    X_train = data[np.ix_(train_ixs, features_ixs)]
    y_train = data[np.ix_(train_ixs, target_ixs[None])]

    X_test = data[np.ix_(test_ixs, features_ixs)]
    y_test = data[np.ix_(test_ixs, target_ixs[None])]

    # Normalise dataset
    xmean, xstd = np.nanmean(X_train, axis=0, keepdims=True), np.nanstd(X_train, axis=0, keepdims=True)
    X_train = (X_train - xmean) / xstd
    X_test = (X_test - xmean) / xstd

    ymean, ystd = np.nanmean(y_train), np.nanstd(y_train)
    y_train = (y_train - ymean) / ystd
    y_test = (y_test - ymean) / ystd

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    train = (X_train, y_train)
    test = (X_test, y_test)

    dataset = train, test
    dataset = jax.tree_map(jnp.array, dataset)
    dataset = jax.tree_map(jnp.nan_to_num, dataset)

    dataset = {
        "train": dataset[0],
        "test": dataset[1],
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
