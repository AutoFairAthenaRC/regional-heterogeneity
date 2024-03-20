import itertools
import effector

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline

class RegionsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, splits, feature_name_prefix="feat_"):
        self.splits = splits
        self.feature_name_prefix = feature_name_prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.to_numpy()
        # feat_mapping <- to how many features each feature is mapped
        feat_mapping = []
        for split in self.splits.values():
            if len(split) == 0:
                feat_mapping.append(1)
            else:
                feat_mapping.append(2**len(split))

        # the enhaned data, without masking
        new_data = []
        for i in range(X.shape[1]):
            new_data.append(np.repeat(X[:, i, np.newaxis], feat_mapping[i], axis=-1))
        new_data = np.concatenate(new_data, axis=-1)

        # create mask, based on splits
        mask = np.ones(new_data.shape)
        new_columns = []
        for feat in range(X.shape[1]):
            # jj = j in the enhanced dataset
            pos = int(np.sum(feat_mapping[:feat]))

            if feat_mapping[feat] == 1:
                new_columns.append("x{}".format(feat))
                continue
            else:
                feat_splits = self.splits["{}{}".format(self.feature_name_prefix, feat)]
                lst = [list(i) for i in itertools.product([0, 1], repeat=len(feat_splits))]
                for ii, bin in enumerate(lst):
                    new_name = "x{} | ".format(feat)
                    init_col_mask = np.ones(new_data.shape[0]) * True
                    for jj, b in enumerate(bin):
                        if b == 0:
                            if feat_splits[jj]["type"] == "cat":
                                init_col_mask = np.logical_and(init_col_mask, X[:, feat_splits[jj]["feature"]] == feat_splits[jj]["position"])
                                # add with two decimals
                                new_name += "x{}={:.2f} & ".format(feat_splits[jj]["feature"], feat_splits[jj]["position"])
                            else:
                                init_col_mask = np.logical_and(init_col_mask, X[:, feat_splits[jj]["feature"]] <= feat_splits[jj]["position"])
                                new_name += "x{}<={:.2f} & ".format(feat_splits[jj]["feature"], feat_splits[jj]["position"])
                        else:
                            if feat_splits[jj]["type"] == "cat":
                                init_col_mask = np.logical_and(init_col_mask, X[:, feat_splits[jj]["feature"]] != feat_splits[jj]["position"])
                                new_name += "x{}!={:.2f} & ".format(feat_splits[jj]["feature"], feat_splits[jj]["position"])
                            else:
                                init_col_mask = np.logical_and(init_col_mask, X[:, feat_splits[jj]["feature"]] > feat_splits[jj]["position"])
                                new_name += "x{}>{:.2f} & ".format(feat_splits[jj]["feature"], feat_splits[jj]["position"])
                    # current position in mask
                    mask[:, pos + ii] = init_col_mask
                    new_columns.append(new_name[:-3])
        self.mask = mask
        self.new_data = new_data * mask
        self.new_names = new_columns
        return self.new_data

def fit_RAM_regression(splits, X_train, y_train, X_test, y_test, evaluate=True, interactions=False, feature_name_prefix="feat_"):
    transf = RegionsTransformer(splits=splits, feature_name_prefix=feature_name_prefix)
    clf = ExplainableBoostingRegressor() if interactions else ExplainableBoostingRegressor(interactions=0)
    model = Pipeline(steps=[
        ("regions", transf),
        ("clf", clf),
    ])
    model.fit(X_train, y_train)
    if evaluate:
        print("RMSE - TEST: ", mean_squared_error(y_test, model.predict(X_test), squared=False))
        print("MAE - TEST", mean_absolute_error(y_test, model.predict(X_test)))

    return model

def fit_RAM_classification(splits, X_train, y_train, X_test, y_test, evaluate=True, interactions=False, feature_name_prefix="feat_"):
    transf = RegionsTransformer(splits=splits, feature_name_prefix=feature_name_prefix)
    clf = ExplainableBoostingClassifier() if interactions else ExplainableBoostingClassifier(interactions=0)
    model = Pipeline(steps=[
        ("regions", transf),
        ("clf", clf),
    ])
    model.fit(X_train, y_train)
    if evaluate:
        print("Accuracy - TEST: ", accuracy_score(y_test, model.predict(X_test)))

    return model



if __name__ == "__main__":
    splits = {
        "feat_0": [],
        "feat_1": [],
        "feat_2": [],
        "feat_3": [],
        "feat_4": [],
        "feat_5": [],
        "feat_6": [],
        "feat_7": [],
        "feat_8": [{"feature": 5, "position": 0, "type": "numeric"}],
        "feat_9": [],
        "feat_10": [],
    }
    # model = fit_RAM(splits, X_train, Y_train, X_test, Y_test)