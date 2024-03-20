from abc import ABC
from ucimlrepo import fetch_ucirepo
from aif360.sklearn.datasets import fetch_compas

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Dataset(ABC):
    pass

def adult():
    adult = fetch_ucirepo(id=2)

    X = adult.data.features # type: ignore
    y = adult.data.targets # type: ignore

    X = X.drop(columns=["fnlwgt", "education"])
    X.replace('?', np.nan, inplace=True)
    X = X.dropna()
    y = y.loc[X.index]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    X["relationship"] = X["relationship"].replace(["Husband", "Wife"], "Married")
    X["hours-per-week"] = pd.cut(
        x=X["hours-per-week"],
        bins=[0.9, 25, 39, 40, 55, 100],
        labels=["PartTime", "MidTime", "FullTime", "OverTime", "BrainDrain"],
    )
    X.age = pd.qcut(X.age, q=5)
    y["income"] = y["income"].apply(lambda x: 0 if x == "<=50K" else 1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    return X, y

def split(X, y, train_size=0.8):
    indices = X.index.tolist()
    np.random.shuffle(indices)
    
    # data split
    train_size = int(train_size * len(X))
    
    X_train = X.iloc[indices[:train_size]].copy().reset_index(drop=True)
    Y_train = y.iloc[indices[:train_size]].copy().reset_index(drop=True)
    X_test = X.iloc[indices[train_size:]].copy().reset_index(drop=True)
    Y_test = y.iloc[indices[train_size:]].copy().reset_index(drop=True)
    
    return X_train, Y_train, X_test, Y_test

def compas():
    X, y = fetch_compas()
    X['target'] = y.values
    X = X.reset_index(drop=True)
    X = X.drop(columns=["age", "c_charge_desc"])
    X["priors_count"] = pd.cut(X["priors_count"], [-0.1, 1, 5, 10, 15, 38])
    X.target.replace("Recidivated", 0, inplace=True)
    X.target.replace("Survived", 1, inplace=True)
    X["age_cat"].replace("Less than 25", "10-25", inplace=True)
    y = X["target"]
    X = X.drop(columns=["target"])

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    return X, y
