import re
import os
import jax
import pickle
import numpy as np
import pandas as pd
from rebayes.datasets import uci_uncertainty_data

def get_subtree(tree, key):
    return jax.tree_map(lambda x: x[key], tree, is_leaf=lambda x: key in x)


def extract_data(files, base_path):
    regexp = re.compile("rank([0-9]+).pkl")
    data_all = {}
    for file in files:
        m = regexp.findall(file)
        if len(m) == 0:
            continue
            rank = 50
        else:
            rank = int(m[0])

        file_path = os.path.join(base_path, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        data_all[rank] = data
    return data_all



def extract_filenames(dataset, base_path):
    files = os.listdir(path)
    files_target = [file for file in files if (dataset in file) and ("pass" not in file)]
    return files_target


def build_df_summary(data, dataset_name):
    """
    Summary over the last-observed value
    """
    agent_last = jax.tree_map(lambda x: x[:, -1], data)
    df_summary = []
    for key in agent_last:
        piece = pd.DataFrame(agent_last[key])

        if key != 1:
            drop_cols = ["fdekf", "vdekf"]
            piece = piece.drop(drop_cols, axis=1)
        if key != 2:
            drop_cols = ["fcekf"]
            piece = piece.drop(drop_cols, axis=1)


        piece = piece.melt()
        piece["rank"] = key
        df_summary.append(piece)
    df_summary = pd.concat(df_summary).dropna(axis=0)
    df_summary = df_summary.query("variable != 'lofi_orth'")

    df_summary.loc[df_summary["variable"] == "fcekf", "rank"] = "full"
    df_summary.loc[df_summary["variable"] == "fdekf", "rank"] = 0
    df_summary.loc[df_summary["variable"] == "vdekf", "rank"] = 0
    df_summary = df_summary.assign(dataset=dataset_name)
    return df_summary


if __name__ == "__main__":
    path = "./output/cross-validation"
    dataset_path = "/home/gerardoduran/documents/external/DropoutUncertaintyExps/UCI_Datasets"
    all_files = os.listdir(path)
    datasets = list(set([f.split("_")[0].split(".")[0] for f in all_files]))

    methods_eval = ["lrvga", "sgd-rb", "lofi"]
    void_datasets = ["protein-tertiary-structure"]
    datasets = [d for d in datasets if d not in void_datasets]

    df_all = []
    for dataset in datasets:
        files_target = extract_filenames(dataset, path)
        data_dataset = extract_data(files_target, path)
        
        
        data = jax.tree_map(
            lambda x: np.atleast_2d(x).mean(axis=0)[-1], data_dataset
        )

        df = []
        for mem, sub in data.items():
            df_part = pd.DataFrame.from_dict(sub, orient="index")
            df_part["memory"] = mem
            df.append(df_part)
        df = pd.concat(df)
        df.index.name = "model"

        df = df.reset_index()
        df = df.query("model in @methods_eval")

        df = df.assign(
            metric=df["test"] / df["running_time"]
        )

        rmin, rmax = df["test"].min(), df["test"].max()

        df["std_test"] = (df["test"] - rmin) / (rmax - rmin)

        df["dataset"] = dataset

        ix = 0
        data_path = os.path.join(dataset_path, dataset, "data")
        res = uci_uncertainty_data.load_data(data_path, ix)

        n_obs, *_ = res["dataset"]["train"][1].shape
        n_obs

        # ~Seconds per datapoint
        df["log_running_time_dp"] = np.log(df["running_time"] / n_obs)
        df_all.append(df)
    df_all = pd.concat(df_all)
    df_all.to_pickle("uci-models-results.pkl")
    print("Done!")
