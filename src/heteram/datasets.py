from abc import ABC, abstractmethod
from ucimlrepo import fetch_ucirepo
from aif360.sklearn.datasets import fetch_compas

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Dataset(ABC):
    @abstractmethod
    def fetch(self):
        pass

    @abstractmethod
    def prep(self):
        pass

    @abstractmethod
    def get_Xy(self):
        pass

class AdultDataset(Dataset):
    def fetch(self):
        adult = fetch_ucirepo(id=2)
        
        self.X = adult.data.features # type: ignore
        self.y = adult.data.targets # type: ignore

        return self
    
    def prep(self):
        X = self.X
        y = self.y

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
        
        self.X = X
        self.y = y
        return self
    
    def get_Xy(self):
        return self.X, self.y

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

class CompasDataset(Dataset):
    def fetch(self):
        self.X, self.y = fetch_compas()
        return self
    
    def prep(self):
        X = self.X
        y = self.y

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
        
        self.X = X
        self.y = y
        return self
    
    def get_Xy(self):
        return self.X, self.y

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

def bike_sharing():
    bike_sharing_dataset = fetch_ucirepo(id=275)
    df = bike_sharing_dataset["data"]["original"]
    df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)

    # Standarize X
    X_df = df.drop(["cnt"], axis=1)
    X_df = (X_df - X_df.mean()) / X_df.std()

    # Standarize Y
    Y_df = df["cnt"]
    Y_df = (Y_df - Y_df.mean()) / Y_df.std()

    return X_df, Y_df
