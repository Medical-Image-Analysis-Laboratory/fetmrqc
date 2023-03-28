from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
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
    GroupScalerSelector,
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
    # METRICS_BASE_CENTER,
    METRICS,
    SCALERS,
    NOISE_FEATURES,
    PCA_FEATURES,
)
from pdb import set_trace
from fetal_brain_qc.qc_evaluation import preprocess as pp
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, SparsePCA
import os
from fetal_brain_qc.qc_evaluation.sacred_helpers import (
    print_dict,
    save_to_csv,
    get_cv,
)

SETTINGS["CAPTURE_MODE"] = "sys"
ex = Experiment("Running nested cross validation on the IQM prediction")
ex.observers.append(MongoObserver())


@ex.config
def config():
    dataset = {
        "dataset_path": "/home/tsanchez/Documents/mial/repositories/mriqc-learn/mriqc_learn/datasets/chuv_bcn.tsv",
        "first_iqm": "centroid",
    }

    experiment = {
        "type": "regression",
        "classification_threshold": None,
        "metrics": "base",
        "scoring": "neg_mae",
    }

    cv = {
        "outer_cv": {"cv": "GroupKFold", "group_by": "sub_ses"},
        "inner_cv": {"cv": "GroupKFold", "group_by": "sub_ses"},
    }

    parameters = {
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


# ToDo: define basic metrics, check that if it's a list then we can use it as well.
@ex.capture(prefix="experiment")
def get_metrics(metrics):
    assert metrics in ["base", "base_center", "full"] + METRICS
    if metrics == "base":
        return METRICS_BASE
    # elif metrics == "base_center":
    #    return METRICS_BASE_CENTER
    elif metrics == "full":
        return METRICS
    elif metrics in METRICS:
        return [metrics]
    else:
        return NotImplementedError


@ex.capture
def read_parameter_grid(experiment, parameters):
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
                    out_grid[k][i] = eval(v)
    return out_grid


@ex.automain
def run(
    dataset,
    experiment,
    cv,
    _config,
):

    is_regression = experiment["type"] == "regression"
    from pathlib import Path
    import pandas as pd

    dataframe = pd.read_csv(
        Path(dataset["dataset_path"]), index_col=None, delimiter=r"\s+"
    )

    # Return the position of the first IQM in the list
    xy_index = dataframe.columns.tolist().index(dataset["first_iqm"])
    dataframe = dataframe.dropna(axis=0)
    cols = dataframe.columns[xy_index:]
    types = {col: float if "nan" not in col else bool for col in cols}
    dataframe = dataframe.astype(types)
    train_x = dataframe[dataframe.columns[xy_index:]].copy()
    train_y = dataframe[dataframe.columns[:xy_index]].copy()
    del dataframe
    metrics_list = get_metrics()

    pipeline = Pipeline(
        steps=[
            (
                "drop_correlated",
                DropCorrelatedFeatures(
                    ignore="group",
                ),
            ),
            ("scaler", pp.GroupScalerSelector(group="group")),
            ("noise_feature", pp.NoiseWinnowFeatSelect()),
            ("pca", PCA(n_components=10)),
            ("model", GradientBoostingRegressor()),
        ]
    )
    params = read_parameter_grid()
    scores = REGRESSION_SCORING if is_regression else CLASSIFICATION_SCORING
    o_cv = get_cv(cv["outer_cv"])
    i_cv = get_cv(cv["inner_cv"])
    print(
        f"Experiment: {experiment['type']} - scoring: {experiment['scoring']}".upper()
        + f" - metrics: {experiment['metrics']}  - outer_group_by: {cv['outer_cv']['group_by']}".upper()
        + "\nUsing parameters:"
    )
    print_dict(params, 2)
    print()

    print(f"CROSS-VALIDATION:\n\tOuter CV: {o_cv}\n\tInner CV: {i_cv}\n")

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring=scores,
        cv=i_cv,
        refit=experiment["scoring"],
        error_score="raise",
    )
    outer_group_by = cv["outer_cv"]["group_by"]
    inner_group_by = cv["inner_cv"]["group_by"]
    outer_groups = train_y[outer_group_by]
    inner_groups = train_y[inner_group_by]
    train_x["group"] = inner_groups

    if not is_regression:
        train_y["rating"] = (
            train_y["rating"] > experiment["classification_threshold"]
        )

    nested_score = cross_validate(
        clf,
        X=train_x[metrics_list + ["group"]],
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
