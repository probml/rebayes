"""
Prepcocessing and data augmentation for the datasets.
"""
import re
import io
import os
import jax
import chex
import zipfile
import requests
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
from jax import vmap




def normalise_dataset(data, target_variable, frac_train, seed, feature_normalise=False, target_normalise=False):
    #TODO: rename to uci_preprocess
    """
    Randomise a dataframe, normalise by column and transform to jax arrays
    """
    data = data.sample(frac=1.0, replace=False, random_state=seed)

    n_train = round(len(data) * frac_train)

    X_train = data.drop(columns=[target_variable]).iloc[:n_train].values
    y_train = data[target_variable].iloc[:n_train].values

    X_test = data.drop(columns=[target_variable]).iloc[n_train:].values
    y_test = data[target_variable].iloc[n_train:].values

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = jnp.maximum(std, 1e-8)

    mean_y = y_train.mean()
    std_y = y_train.std()

    if feature_normalise:
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if target_normalise:
        y_train = (y_train - mean_y) / std_y
        y_test = (y_test - mean_y) / std_y

    # Convert to jax arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test)

    train = (X_train, y_train)
    test = (X_test, y_test)
    return train, test


def load_uci_wine_regression(color="all", frac_train=0.8, include_color=False, seed=314, normalise=True):
    """
    https://archive.ics.uci.edu/ml/datasets/wine+quality
    """
    target_variable = "quality"
    url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{color}.csv"


    if (color == "red") | (color == "white"):
        colorv = 0 if color == "red" else 1
        url = url_base.format(color=color)
        data = (
            pd.read_csv(url, sep=";")
            .assign(color=colorv)
        )
    elif color == "all":
        data_red = load_uci_wine_regression(color="red", include_color=True, normalise=False)
        data_white = load_uci_wine_regression(color="white", include_color=True, normalise=False)

        data = pd.concat([data_red, data_white], axis=0)

    if not include_color:
        data = data.drop(columns=["color"])

    if normalise:
        data = normalise_dataset(data, target_variable, frac_train, seed)

    return data


def load_uci_naval(target_variable="ship_speed", frac_train=0.8, seed=314):
    """
    http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants

    Target column is one of the following (default is "ship_speed"):
    * lever_position
    * ship_speed
    * gas_turbine_shaft_torque
    * gas_turbine_rate_of_revolutions
    * gas_generator_rate_of_revolutions
    * starboard_propeller_torque
    * port_propeller_torque
    * hp_turbine_exit_temperature
    * gt_compressor_inlet_air_temperature
    * gt_compressor_outlet_air_temperature
    * hp_turbine_exit_pressure
    * gt_compressor_inlet_air_pressure
    * gt_compressor_outlet_air_pressure
    * gas_turbine_exhaust_gas_pressure
    * turbine_injecton_control
    * fuel_flow
    * gt_compressor_decay_state_coefficient
    * gt_turbine_decay_state_coefficient
    """
    file_target = "UCI CBM Dataset/data.txt"
    file_features = "UCI CBM Dataset/Features.txt"
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip"

    r = requests.get(url)
    file = io.BytesIO(r.content)
    with zipfile.ZipFile(file) as zip:
        with zip.open(file_features) as f:
            features = f.read().decode("utf-8")
            regexp = re.compile("[0-9]{1,2} - ([\w\s]+)")
            features = regexp.findall(features)
            features = [f.lower().rstrip().replace(" ", "_") for f in features]

        with zip.open(file_target) as f:
            data = f.read().decode("utf-8")
            data = io.StringIO(data)
            data = pd.read_csv(data, sep="\s+", header=None, engine="python")
            data.columns = features

    data = normalise_dataset(data, target_variable, frac_train, seed)
    return data


def load_uci_kin8nm(frac_train=0.8, seed=314):
    url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"
    target_variable = "y"
    data = pd.read_csv(url)

    data = normalise_dataset(data, target_variable, frac_train, seed)    
    return data


def load_uci_power(frac_train=0.8, seed=314):
    """
    https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    file_target = "CCPP/Folds5x2_pp.xlsx"
    target_variable = "PE"

    r = requests.get(url)
    file = io.BytesIO(r.content)
    with zipfile.ZipFile(file) as zip:
        with zip.open(file_target) as f:
            data = io.BytesIO(f.read())
            data = pd.read_excel(data)
        
    data = normalise_dataset(data, target_variable, frac_train, seed)
    return data


def load_uci_protein(frac_train=0.8, seed=314):
    """
    https://github.com/yaringal/DropoutUncertaintyExps/tree/master/UCI_Datasets/protein-tertiary-structure
    """
    target_variable = 9
    url = (
        "https://raw.githubusercontent.com/yaringal/"
        "DropoutUncertaintyExps/master/UCI_Datasets/"
        "protein-tertiary-structure/data/data.txt"
    )
    data = pd.read_csv(url, header=None, sep=" ")
    data = normalise_dataset(data, target_variable, frac_train, seed)
    return data


def load_uci_spam(frac_train=0.8, seed=314):
    """
    https://archive.ics.uci.edu/ml/datasets/spambase
    """
    target_variable = 57
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    data = pd.read_csv(url, header=None)
    data = normalise_dataset(data, target_variable, frac_train, seed, target_normalise=False)
    return data

