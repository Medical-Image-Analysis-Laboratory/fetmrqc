from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
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
    GroupQuantileTransformer,
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
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
)
import os
from fetal_brain_qc.qc_evaluation.sacred_helpers import (
    print_dict,
    save_to_csv,
    get_cv,
)
from pathlib import Path
import pandas as pd
import ast
from sklearn.model_selection import RandomizedSearchCV

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
        "randomized_inner_cv": False,
        "n_randomized": 50,
        "transform_target": False,
        "tt_dist": "uniform",
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
    """
    Returns a list of metrics based on the input string.

    Args:
        metrics (str): A string indicating the type of metrics to return. Valid options are:
            - 'base': returns the base metrics.
            - 'base_center': returns the base center metrics (currently not implemented).
            - 'full': returns all metrics.
            - 'base_seg': returns the base metrics and the segmentation metrics.
            - 'seg': returns only the segmentation metrics.
            - 'full_seg': returns all metrics and the segmentation metrics.
            - any metric name: returns a list with the specified metric only.

    Returns:
        list: A list of metric names.

    Raises:
        AssertionError: If the input string is not a valid option.
    """
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


def add_to_param(param, input):
    param = param.split(")")[0]
    param += ", " if param[-1] != "(" else " "
    param += f"{input})"
    return param


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
    print(parameters)
    out_grid = copy.deepcopy(parameters)
    for k, values in parameters.items():
        if k in ["pca", "noise_feature", "scaler__scaler", "model"]:
            for i, v in enumerate(values):
                print(v)
                v_str = v.split("(")[0]
                print(v_str)
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
                    if (
                        k == "model"
                        and rng is not None
                        and hasattr(eval(v), "random_state")
                    ):
                        # RNG control: Adding the random state to the model
                        # for reproducibility
                        v = add_to_param(v, "random_state=rng")
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

    # Transforming the target: using the quantile transformer
    # to transform the target to a uniform or random distribution
    if experiment["transform_target"]:
        params = {
            "model__regressor__" + k.split("model__")[1]
            if "model__" in k
            else k: v
            for k, v in params.items()
        }
        params["model"] = [
            TransformedTargetRegressor(
                p,
                transformer=QuantileTransformer(
                    output_distribution=experiment["tt_dist"], n_quantiles=250
                ),
            )
            for p in params["model"]
        ]

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

    # Inner CV: either full grid search or randomized search
    if experiment["randomized_inner_cv"]:
        inner_cv_opt = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            scoring=scores,
            n_iter=experiment["n_randomized"],
            cv=i_cv,
            random_state=rng,
            refit=experiment["scoring"],
        )
    else:
        inner_cv_opt = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring=scores,
            cv=i_cv,
            refit=experiment["scoring"],
            error_score="raise",
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

    ### Convert all the column which have a dtype of object to bool, keep the ones that are float as float
    train_x = train_x.astype(
        {
            col: bool
            for col in train_x.columns
            if train_x[col].dtype == "object"
        }
    )
    nested_score = cross_validate(
        inner_cv_opt,
        X=train_x[m],
        y=train_y["rating"],
        cv=o_cv,
        scoring=scores,
        groups=outer_groups,
        n_jobs=5,
        fit_params={"groups": inner_groups},
        return_train_score=True,
        return_estimator=True,
        error_score="raise",
    )

    outer_groups_list = []
    for i, (train_index, test_index) in enumerate(
        o_cv.split(train_x[m], train_y["rating"], outer_groups)
    ):
        group = np.unique(outer_groups[test_index].to_numpy()).tolist()
        outer_groups_list.append(group)
    nested_score["test_outer_groups"] = outer_groups_list
    # nested_score["inner_groups"] = inner_groups
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

    print(nested_score.keys(), nested_score["test_outer_groups"])

    print("FINAL RESULTS")
    for k, v in nested_score.items():
        if "test" in k and "groups" not in k:
            ex.log_scalar(f"{k}_mean", v.mean())
            ex.log_scalar(f"{k}_std", v.std())
            print(f"\t{k:20} = {v.mean():6.3f} +- {v.std():5.3f}")

    out_file = "nested_score.dill"
    with open(out_file, "wb") as f:
        dill.dump(nested_score, f)

    ex.add_artifact(out_file)

    save_to_csv(ex, _config, nested_score, metrics_list)
    os.remove(out_file)
