from pprint import pprint
from argparse import ArgumentParser

from sklearn.metrics import mean_absolute_error, mean_squared_error
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import effector
from effector.binning_methods import Greedy

from datasets import adult, compas, bike_sharing
from data_utils import split
from models import bike_sharing_NN_fit, wrap_tf_model, adult_NN_fit, compas_NN_fit
from build_ram import fit_RAM_classification, fit_RAM_regression

def splits_PDP(X_train, model_forward, feature_names):
    regional_pdp = effector.RegionalPDP(
        data=X_train.to_numpy(),
        model=model_forward,
        cat_limit=10,
        feature_names=feature_names,
        nof_instances="all"
    )

    regional_pdp.fit(
        features="all",
        heter_small_enough=0.1,
        heter_pcg_drop_thres=0.1,
        max_depth=2,
        nof_candidate_splits_for_numerical=5,
        min_points_per_subregion=10,
        candidate_conditioning_features="all",
        split_categorical_features=True,
        nof_instances=1000
    )

    splits = {feat_name: partitioner.important_splits for feat_name, partitioner in regional_pdp.partitioners.items()}
    return splits

def splits_RHALE(X_train, model_forward, model_jac, feature_names):
    regional_rhale = effector.RegionalRHALE(
        data=X_train.to_numpy(),
        model=model_forward,
        model_jac=model_jac,
        cat_limit=45,
        feature_names=feature_names,
        nof_instances="all"
    )
    regional_rhale.fit(
        features="all",
        heter_small_enough=0.1,
        heter_pcg_drop_thres=0.1,
        binning_method=Greedy(init_nof_bins=100, min_points_per_bin=10, discount=1., cat_limit=45),
        max_depth=2,
        nof_candidate_splits_for_numerical=10,
        min_points_per_subregion=5,
        candidate_conditioning_features="all",
        split_categorical_features=True,
    )

    splits = {feat_name: partitioner.important_splits for feat_name, partitioner in regional_rhale.partitioners.items()}
    return splits

def run_adult():
    X, y = adult()
    
    cols = X.columns
    means = X[cols].mean()
    stds = X[cols].std()
    X[cols] = (X[cols] - means) / stds

    X_train, Y_train, X_test, Y_test = split(X, y)

    model = adult_NN_fit(X_train, Y_train)

    model_forward, model_jac = wrap_tf_model(model=model)

    feature_names = X.columns.tolist()

    splits = splits_PDP(X_train, model_forward, feature_names)
    ram_PDP = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=False, feature_name_prefix="feature_")
    ra2m_PDP = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=True, feature_name_prefix="feature_")

    splits = splits_RHALE(X_train, model_forward, model_jac, feature_names)
    ram_RHALE = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=False, feature_name_prefix="feature_")
    ra2m_RHALE = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=True, feature_name_prefix="feature_")

    gam = ExplainableBoostingClassifier(interactions=0)
    gam.fit(X_train, Y_train)
    ga2m = ExplainableBoostingClassifier()
    ga2m.fit(X_train, Y_train)

    return {
        "ram_PDP": ram_PDP.score(X_test, Y_test.astype(int)),
        "ra2m_PDP": ra2m_PDP.score(X_test, Y_test.astype(int)),
        "ram_RHALE": ram_RHALE.score(X_test, Y_test.astype(int)),
        "ra2m_RHALE": ra2m_RHALE.score(X_test, Y_test.astype(int)),
        "gam": gam.score(X_test, Y_test.astype(int)),
        "ga2m": ga2m.score(X_test, Y_test.astype(int)),
        "dnn": model.evaluate(X_test, Y_test, verbose=0)[1],
    }

def run_compas():
    X, y = compas()

    cols = X.columns
    means = X[cols].mean()
    stds = X[cols].std()
    X[cols] = (X[cols] - means) / stds

    X_train, Y_train, X_test, Y_test = split(X, y)

    model = compas_NN_fit(X_train, Y_train)

    model_forward, model_jac = wrap_tf_model(model=model)

    feature_names = X.columns.tolist()

    splits = splits_PDP(X_train, model_forward, feature_names)
    ram_PDP = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=False, feature_name_prefix="feature_")
    ra2m_PDP = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=True, feature_name_prefix="feature_")

    splits = splits_RHALE(X_train, model_forward, model_jac, feature_names)
    ram_RHALE = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=False, feature_name_prefix="feature_")
    ra2m_RHALE = fit_RAM_classification(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=True, feature_name_prefix="feature_")

    gam = ExplainableBoostingClassifier(interactions=0)
    gam.fit(X_train, Y_train)
    ga2m = ExplainableBoostingClassifier()
    ga2m.fit(X_train, Y_train)

    return {
        "ram_PDP": ram_PDP.score(X_test, Y_test.astype(str)),
        "ra2m_PDP": ra2m_PDP.score(X_test, Y_test.astype(str)),
        "ram_RHALE": ram_RHALE.score(X_test, Y_test.astype(str)),
        "ra2m_RHALE": ra2m_RHALE.score(X_test, Y_test.astype(str)),
        "gam": gam.score(X_test, Y_test.astype(str)),
        "ga2m": ga2m.score(X_test, Y_test.astype(str)),
        "dnn": model.evaluate(X_test, Y_test, verbose=0)[1],
    }

def run_bike_sharing():
    X, y = bike_sharing()

    X_train, Y_train, X_test, Y_test = split(X, y)

    model = bike_sharing_NN_fit(X_train, Y_train)

    model_forward, model_jac = wrap_tf_model(model=model)

    feature_names = X.columns.tolist()

    splits = splits_PDP(X_train, model_forward, feature_names)
    ram_PDP = fit_RAM_regression(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=False, feature_name_prefix="feature_")
    ram_PDP_preds = ram_PDP.predict(X_test)
    ra2m_PDP = fit_RAM_regression(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=True, feature_name_prefix="feature_")
    ra2m_PDP_preds = ra2m_PDP.predict(X_test)

    splits = splits_RHALE(X_train, model_forward, model_jac, feature_names)
    ram_RHALE = fit_RAM_regression(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=False, feature_name_prefix="feature_")
    ram_RHALE_preds = ram_RHALE.predict(X_test)
    ra2m_RHALE = fit_RAM_regression(splits, X_train, Y_train, X_test, Y_test, evaluate=False, interactions=True, feature_name_prefix="feature_")
    ra2m_RHALE_preds = ra2m_RHALE.predict(X_test)

    gam = ExplainableBoostingRegressor(interactions=0)
    gam.fit(X_train, Y_train)
    gam_preds = gam.predict(X_test)
    ga2m = ExplainableBoostingRegressor()
    ga2m.fit(X_train, Y_train)
    ga2m_preds = ga2m.predict(X_test)

    return {
        "ram_PDP": {
            "mae": mean_absolute_error(Y_test, ram_PDP_preds),
            "rmse": mean_squared_error(Y_test, ram_PDP_preds, squared=False)
        },
        "ra2m_PDP": {
            "mae": mean_absolute_error(Y_test, ra2m_PDP_preds),
            "rmse": mean_squared_error(Y_test, ra2m_PDP_preds, squared=False)
        },
        "ram_RHALE": {
            "mae": mean_absolute_error(Y_test, ram_RHALE_preds),
            "rmse": mean_squared_error(Y_test, ram_RHALE_preds, squared=False)
        },
        "ra2m_RHALE": {
            "mae": mean_absolute_error(Y_test, ra2m_RHALE_preds),
            "rmse": mean_squared_error(Y_test, ra2m_RHALE_preds, squared=False)
        },
        "gam": {
            "mae": mean_absolute_error(Y_test, gam_preds),
            "rmse": mean_squared_error(Y_test, gam_preds, squared=False)
        },
        "ga2m": {
            "mae": mean_absolute_error(Y_test, ga2m_preds),
            "rmse": mean_squared_error(Y_test, ga2m_preds, squared=False)
        },
        "dnn": {
            "mae": model.evaluate(X_test, Y_test, verbose=0)[1],
            "rmse": model.evaluate(X_test, Y_test, verbose=0)[2],
        },
    }

def main():
    arg_parser = ArgumentParser(description="Builds and Evaluates RAMs for several datasets.")
    arg_parser.add_argument("--datasets", nargs="*")
    args = arg_parser.parse_args()

    if args.datasets is None or "adult" in args.datasets:
        print("Adult dataset:")
        pprint(run_adult())
    if args.datasets is None or "compas" in args.datasets:
        print("Compas dataset:")
        pprint(run_compas())
    if args.datasets is None or "bike-sharing" in args.datasets:
        print("Bike Sharing dataset:")
        pprint(run_bike_sharing())

if __name__ == "__main__":
    main()
