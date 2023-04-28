from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fetal_brain_qc.qc_evaluation.qc_evaluation import (
    REGRESSION_SCORING,
    REGRESSION_MODELS,
    CLASSIFICATION_MODELS,
    CLASSIFICATION_SCORING,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    GroupShuffleSplit,
)
from fetal_brain_qc.qc_evaluation.load_dataset import load_dataset

from sklearn.model_selection import cross_validate, GroupKFold
import numpy as np
from fetal_brain_qc.qc_evaluation import preprocess as pp
from fetal_brain_qc.qc_evaluation.preprocess import (
    GroupRobustScaler,
    GroupStandardScaler,
    GroupScalerSelector,
    PassThroughScaler,
    NoiseWinnowFeatSelect,
    DropCorrelatedFeatures,
)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from xgboost import XGBRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import (
    LinearRegression,
    # ElasticNet,
    Ridge,
    Lasso,
    HuberRegressor,
    QuantileRegressor,
    LogisticRegression,
    RidgeClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import pickle
import dill
from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import MongoObserver
import copy
import inspect
from fetal_brain_qc.qc_evaluation import (
    VALID_EXP,
    METRICS_BASE,
    METRICS_SEG,
    # METRICS_BASE_CENTER,
    METRICS,
    SCALERS,
    NOISE_FEATURES,
    PCA_FEATURES,
)
from pdb import set_trace
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
from fetal_brain_qc.qc_evaluation.sacred_helpers import (
    print_dict,
    save_to_csv,
    get_cv,
)
from pathlib import Path
import pandas as pd
import ast

SETTINGS["CAPTURE_MODE"] = "sys"
ex = Experiment("Running nested cross validation on the IQM prediction")
ex.observers.append(MongoObserver())


@ex.config
def config():

    dataset = {  # noqa: F841
        "dataset_path": "/home/tsanchez/Documents/mial/repositories/mriqc-learn/mriqc_learn/datasets/chuv_bcn.tsv",
        "first_iqm": "centroid",
    }

    experiment = {
        "type": "regression",
        "classification_threshold": None,
        "metrics": "base",
        "scoring": "neg_mae",
        "normalization_feature": None,
    }

    cv = {
        "outer_cv": {"cv": "GroupKFold", "group_by": "sub_ses"},
        "inner_cv": {"cv": "GroupKFold", "group_by": "sub_ses"},
    }

    parameters = {  # noqa: F841
        "drop_correlated__threshold": [0.95, 1.0],
        "pca": ["passthrough"],  # PCA()
        "noise_feature": ["passthrough"],  # NoiseWinnowFeatSelect()
        "scaler__scaler": ["StandardScaler()", "RobustScaler()"],
        "model": ["LinearRegression()"],
    }
    name = f"{experiment['type']}_{experiment['scoring']}_{experiment['metrics']}_{cv['outer_cv']['group_by']}"
    ex.path = name


@ex.capture(prefix="experiment")
def check_entries(type, metrics, scoring):
    assert (
        type in VALID_EXP
    ), f"Invalid exp_type={type}, please choose between {VALID_EXP}"
    if type == "regression":
        assert (
            scoring in REGRESSION_SCORING.keys()
        ), f"Invalid eval_score={scoring}, please choose between {REGRESSION_SCORING.keys()}"
    else:
        assert (
            scoring in CLASSIFICATION_SCORING.keys()
        ), f"Invalid eval_score={scoring}, please choose between {CLASSIFICATION_SCORING.keys()}"


def get_metrics(metrics):
    assert (
        metrics
        in ["base", "base_center", "base_seg", "full", "full_seg", "seg"]
        + METRICS
    )
    if metrics == "base":
        return METRICS_BASE
    # elif metrics == "base_center":
    #    return METRICS_BASE_CENTER
    elif metrics == "full":
        return METRICS
    elif metrics == "base_seg":
        return METRICS_BASE + METRICS_SEG
    elif metrics == "seg":
        return METRICS_SEG
    elif metrics == "full_seg":
        return METRICS + METRICS_SEG
    elif metrics in METRICS:
        return [metrics]
    else:
        return NotImplementedError


@ex.capture
def read_parameter_grid(experiment, parameters, rng=None):
    exp_type = experiment["type"]
    MODELS = (
        REGRESSION_MODELS
        if exp_type == "regression"
        else CLASSIFICATION_MODELS
    )
    keys_to_eval = {
        "model": MODELS,
        "scaler__scaler": SCALERS,
        "noise_feature": NOISE_FEATURES,
        "pca": PCA_FEATURES,
    }

    out_grid = copy.deepcopy(parameters)
    for k, values in parameters.items():
        if k in ["pca", "noise_feature", "scaler__scaler", "model"]:
            for i, v in enumerate(values):
                v_str = v.split("(")[0]
                if v_str in keys_to_eval[k]:
                    assert v == "passthrough"
                    out_grid[k][i] = v
                else:
                    keys = [
                        x.__class__.__name__
                        for x in keys_to_eval[k]
                        if not isinstance(x, str)
                    ]
                    assert v_str in keys, (
                        f"ERROR: function {v_str} from {k} in parameter_grid is not supported. "
                        f"Please choose among the following options: {keys_to_eval[k]}"
                    )
                    if k == "model" and rng is not None:
                        # RNG control: Adding the random state to the model
                        # for reproducibility
                        v = v.split(")")[0]
                        v += ", " if v[-1] != "(" else " "
                        v += "random_state=rng)"
                    out_grid[k][i] = eval(v)
                print("Reading parameters.")
    return out_grid


def run_experiment(dataset, experiment, cv, parameters, seed):
    rng = np.random.RandomState(seed)
    is_regression = experiment["type"] == "regression"

    if dataset["dataset_path"].endswith(".tsv"):

        dataframe = pd.read_csv(
            Path(dataset["dataset_path"]), index_col=None, delimiter=r"\s+"
        )
    elif dataset["dataset_path"].endswith(".csv"):
        dataframe = pd.read_csv(Path(dataset["dataset_path"]), index_col=None)
    else:
        raise ValueError("Dataset should be a csv or tsv file")

    # Return the position of the first IQM in the list
    xy_index = dataframe.columns.tolist().index(dataset["first_iqm"])
    dataframe = dataframe.dropna(axis=0)
    cols = dataframe.columns[xy_index:]
    types = {col: float if "nan" not in col else bool for col in cols}
    dataframe = dataframe.astype(types)
    train_x = dataframe[dataframe.columns[xy_index:]].copy()
    train_y = dataframe[dataframe.columns[:xy_index]].copy()
    norm_feature = (
        None
        if experiment["normalization_feature"] is None
        or experiment["normalization_feature"].lower() == "none"
        else experiment["normalization_feature"]
    )
    feature = dataframe[norm_feature] if norm_feature is not None else None

    if norm_feature == "im_size_vx_size":
        feature = (feature > 3).astype(int)
    del dataframe
    metrics_list = get_metrics(metrics=experiment["metrics"])

    o_cv = get_cv(cv["outer_cv"], rng=rng)
    i_cv = get_cv(cv["inner_cv"], rng=rng)
    outer_group_by = cv["outer_cv"]["group_by"]
    inner_group_by = cv["inner_cv"]["group_by"]
    outer_groups = train_y[outer_group_by]
    inner_groups = train_y[inner_group_by]

    if norm_feature is not None:
        train_x[norm_feature] = feature
        groupby = norm_feature
        print(f"Group is {norm_feature}")
    else:
        # train_x[inner_group_by] = inner_groups
        groupby = None  # inner_group_by

    pipeline = Pipeline(
        steps=[
            ("scaler", pp.GroupScalerSelector(group=groupby)),
            (
                "drop_correlated",
                DropCorrelatedFeatures(
                    ignore=None,
                ),
            ),
            ("noise_feature", pp.NoiseWinnowFeatSelect()),
            ("pca", PCA(n_components=10)),
            ("model", GradientBoostingRegressor()),
        ]
    )
    params = read_parameter_grid(experiment, parameters, rng)
    # params.update(
    #    {'model__bootstrap': [True, False],
    #     'model__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #     'model__max_features': ['log2', 'sqrt'],
    #     'model__min_samples_leaf': [1, 2, 4],
    #     'model__min_samples_split': [2, 5, 10],
    #     'model__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    # )
    params.update(
        {
            "model__n_estimators": [0],
            "model__alpha": [1e-08, 1e-4, 1e0, 100.0],
            "model__colsample_bylevel": [0.5, 0.75, 1.0],
            "model__colsample_bytree": [0.5, 0.75, 1.0],
            "model__gamma": [1e-8, 1e-3, 1e-1, 100],
            "model__lambda": [1e-8, 1e-3, 1e-1, 100],
            "model__learning_rate": [1e-5, 1e-3, 1e-1, 100],
            "model__max_depth": [3, 5, 7, 10],
            "model__min_child_weight": [1e-8, 1e-2, 1e2, 1e5],
            "model__subsample": [0.5, 0.75, 1.0],
        }
    )

    if norm_feature is not None:
        if not any(
            [
                isinstance(p, (GroupStandardScaler, GroupRobustScaler))
                for p in params["scaler__scaler"]
            ]
        ):
            print(
                "WARNING: normalization_feature is set but no Group based scaler is selected. Ignoring normalization_feature"
            )
    scores = REGRESSION_SCORING if is_regression else CLASSIFICATION_SCORING

    print(
        f"Experiment: {experiment['type']} - scoring: {experiment['scoring']}".upper()
        + f" - metrics: {experiment['metrics']}  - outer_group_by: {cv['outer_cv']['group_by']}".upper()
        + "\nUsing parameters:"
    )
    print_dict(params, 2)
    print()

    print(f"CROSS-VALIDATION:\n\tOuter CV: {o_cv}\n\tInner CV: {i_cv}\n")
    # inner_cv_opt = GridSearchCV(
    #    estimator=pipeline,
    #    param_grid=params,
    #    scoring=scores,
    #    cv=i_cv,
    #    refit=experiment["scoring"],
    #    error_score="raise",
    # )
    from sklearn.model_selection import RandomizedSearchCV

    inner_cv_opt = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=50,
        cv=i_cv,
        verbose=2,
        random_state=rng,
        n_jobs=-1,
    )
    if not is_regression:
        train_y["rating"] = (
            train_y["rating"] > experiment["classification_threshold"]
        )
    m = (
        metrics_list + [groupby]
        if groupby is not None and groupby not in metrics_list
        else metrics_list
    )
    nested_score = cross_validate(
        inner_cv_opt,
        X=train_x[m],
        y=train_y["rating"],
        cv=o_cv,
        scoring=scores,
        groups=outer_groups,
        n_jobs=16,
        fit_params={"groups": inner_groups},
        return_train_score=True,
        return_estimator=True,
        error_score="raise",
    )
    return nested_score, metrics_list


@ex.automain
def run(
    dataset,
    experiment,
    cv,
    parameters,
    _config,
):

    nested_score, metrics_list = run_experiment(
        dataset, experiment, cv, parameters, _config["seed"]
    )
    print("FINAL RESULTS")
    for k, v in nested_score.items():
        if "test" in k:
            ex.log_scalar(f"{k}_mean", v.mean())
            ex.log_scalar(f"{k}_std", v.std())
            print(f"\t{k:20} = {v.mean():6.3f} +- {v.std():5.3f}")

    out_file = "nested_score.dill"
    with open(out_file, "wb") as f:
        dill.dump(nested_score, f)

    ex.add_artifact(out_file)

    save_to_csv(ex, _config, nested_score, metrics_list)
    os.remove(out_file)
