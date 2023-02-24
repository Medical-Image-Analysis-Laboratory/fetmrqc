from sklearn.decomposition import PCA, SparsePCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
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
    PassThroughScaler,
    GroupRobustScaler,
    GroupStandardScaler,
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

from sacred import Experiment
from sacred import SETTINGS

SCALERS = [
    StandardScaler(),
    RobustScaler(),
    PassThroughScaler(),
    GroupRobustScaler(),
    GroupStandardScaler(),
]
NOISE_FEATURES = ["passthrough", pp.NoiseWinnowFeatSelect()]
PCA_FEATURES = ["passthrough", PCA(), SparsePCA()]

SETTINGS["CAPTURE_MODE"] = "sys"
ex = Experiment("Running nested cross validation on the IQM prediction")


class PrintDebug(BaseEstimator, TransformerMixin):
    """Report to the log current columns."""

    def fit(self, X, y=None):
        print(X)
        return self

    def transform(self, X, y=None):
        return X


VALID_EXP = ["regression", "classification"]
METRICS_BASE = [
    "dl_slice_iqa_pos_only_full",
    "dl_stack_iqa_full",
    "rank_error_full",
    "mask_volume",
    "centroid_full",
]
METRICS_BASE += [m + "_nan" for m in METRICS_BASE]
METRICS_BASE_CENTER = [
    "centroid",
    "centroid_full",
    "dl_slice_iqa",
    "dl_slice_iqa_pos_only_full",
    "dl_stack_iqa_full",
    "rank_error",
    "rank_error_center",
    "rank_error_full",
    "mask_volume",
]
METRICS_BASE_CENTER += [m + "_nan" for m in METRICS_BASE_CENTER]
METRICS = [
    "centroid",
    "centroid_full",
    "bias",
    "bias_full",
    "bias_full_not_mask",
    "dilate_erode_mask",
    "dilate_erode_mask_full",
    "dl_slice_iqa",
    "dl_slice_iqa_cropped",
    "dl_slice_iqa_full",
    "dl_slice_iqa_full_cropped",
    "dl_slice_iqa_pos_only_full",
    "dl_stack_iqa_full",
    "filter_laplace",
    "filter_laplace_full",
    "filter_laplace_mask",
    "filter_laplace_mask_full",
    "filter_sobel",
    "filter_sobel_full",
    "filter_sobel_mask",
    "filter_sobel_mask_full",
    "joint_entropy",
    "joint_entropy_full",
    "joint_entropy_intersection",
    "joint_entropy_median",
    "joint_entropy_window",
    "kurtosis",
    "kurtosis_full",
    "mae",
    "mae_window",
    "mask_volume",
    "mean",
    "mean_full",
    "median",
    "median_full",
    "mi",
    "mi_full",
    "mi_intersection",
    "mi_median",
    "mi_window",
    "ncc",
    "ncc_full",
    "ncc_intersection",
    "ncc_median",
    "ncc_window",
    "nmae",
    "nmae_window",
    "nmi",
    "nmi_full",
    "nmi_intersection",
    "nmi_median",
    "nmi_window",
    "nrmse",
    "nrmse_window",
    "percentile_5",
    "percentile_5_full",
    "percentile_95",
    "percentile_95_full",
    "psnr",
    "psnr_window",
    "rank_error",
    "rank_error_center",
    "rank_error_center_relative",
    "rank_error_full",
    "rank_error_full_cropped_relative",
    "rmse",
    "rmse_window",
    "shannon_entropy",
    "shannon_entropy_full",
    "ssim",
    "ssim_window",
    "std",
    "std_full",
    "variation",
    "variation_full",
]
METRICS += [m + "_nan" for m in METRICS]


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
        "outer_cv": {"cv": "GroupKFold", "n_splits": 2, "group_by": "sub_ses"},
        "inner_cv": {"cv": "GroupKFold", "n_splits": 2, "group_by": "sub_ses"},
    }

    parameters = {
        "drop_correlated__threshold": [0.95, 1.0],
        "pca": ["passthrough"],  # PCA()
        "noise_feature": ["passthrough"],  # NoiseWinnowFeatSelect()
        "scaler__scaler": ["StandardScaler()", "RobustScaler()"],
        "model": ["LinearRegression()"],
    }


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
    assert metrics in ["base", "base_center", "full"]
    if metrics == "base":
        return METRICS_BASE
    elif metrics == "base_center":
        return METRICS_BASE_CENTER
    elif metrics == "full":
        return METRICS
    else:
        return NotImplementedError


import copy
import inspect


def get_cv(cv_dict):
    cv_copy = copy.deepcopy(cv_dict)
    cv = cv_copy.pop("cv")
    group_by = cv_copy.pop("group_by", None)
    valid_cv = [
        "GroupKFold",
        "GroupShuffleSplit",
    ]
    assert cv in valid_cv, f"The only valid CV splitters are {valid_cv}"
    # Get class from globals and create an instance
    cv_func = globals()[cv]
    argspec = inspect.getfullargspec(cv_func).args
    argspec.remove("self")
    for k in cv_copy.keys():
        if k not in argspec:
            raise RuntimeError(
                f"Invalid key in {cv_dict}. {k} is not a key from {cv_func} "
            )
    cv_model = cv_func(**cv_copy)
    return cv_model


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


def print_dict(d, indent=0):
    sp = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{sp}{k}:")
            print_dict(v, indent + 2)
        elif isinstance(v, list):
            print(f"{sp}{k}:")
            for el in v:
                print(f"{sp}- {el}")
        else:
            print(f"{sp}{k}: {v}")


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
            ("scaler", GroupScalerSelector(group="group")),
            ("noise_feature", NoiseWinnowFeatSelect()),
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
        + "\nUsing parameters:"
    )
    print_dict(params, 2)
    print()

    print(f"CROSS-VALIDATION:\n\tOuter CV: {o_cv}\n\tInner CV: {i_cv}\n")
    print(scores, i_cv, pipeline, params)
    from pdb import set_trace

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
    # print(nested_score)
    import pickle

    out_file = "nested_score.npz"
    with open(out_file, "wb") as f:
        pickle.dump(nested_score, f)
    ex.add_artifact(out_file)
    import os

    os.remove(out_file)
