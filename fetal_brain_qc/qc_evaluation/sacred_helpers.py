import copy
import os
import inspect
import pandas as pd
from pdb import set_trace


def print_dict(d, indent=0):
    """Print a dictionary like a yaml file"""
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


def get_cv(cv_dict, rng=None):
    """From a dictionary of attributed of cross validation, get a cross validation object
    The dictionary must have `{"cv": cv_type, "cv_param1":val1, ...}` and the evaluation will be
    `cv_type(cv_param1=val1, ...)`
    """
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from .model_selection import CustomStratifiedGroupKFold

    cv_copy = copy.deepcopy(cv_dict)
    cv = cv_copy.pop("cv")
    group_by = cv_copy.pop("group_by", None)
    valid_cv = [
        "GroupKFold",
        "GroupShuffleSplit",
        "CustomStratifiedGroupKFold",
    ]
    assert cv in valid_cv, f"The only valid CV splitters are {valid_cv}"
    # Get class from globals and create an instance
    cv_func = locals()[cv]
    argspec = inspect.getfullargspec(cv_func)
    args = argspec.args
    args.remove("self")
    if hasattr(argspec, "kwonlyargs"):
        args += argspec.kwonlyargs
    for k in cv_copy.keys():
        if k not in args:
            raise RuntimeError(
                f"Invalid key in {cv_dict}. {k} is not a key from {cv_func} "
            )
    if rng is not None and cv in [
        "GroupShuffleSplit",
        "CustomStratifiedGroupKFold",
    ]:
        cv_copy["random_state"] = rng
    cv_model = cv_func(**cv_copy)
    return cv_model


def get_feature_importance(estimator, metrics):
    drop_correlated = estimator["drop_correlated"]
    pca = estimator["pca"]
    drop_noise = estimator["noise_feature"]
    dropped = []
    if drop_correlated != "passthrough":
        dropped += drop_correlated.drop

    if drop_noise != "passthrough":
        dropped += drop_noise.drop
    retained_features = [f for f in metrics if f not in dropped]
    model = estimator["model"]

    if hasattr(model, "coef_"):
        feature_imp = model.coef_.flatten()
    elif hasattr(model, "feature_importances_"):
        feature_imp = model.feature_importances_.flatten()
    else:
        raise RuntimeError(f"Unknown feature importance for model={model}")
    if pca == "passthrough":
        feature_imp = {k: v for k, v in zip(retained_features, feature_imp)}
    else:
        feature_imp = {f"feature_{i+1}": v for i, v in enumerate(feature_imp)}
        feature_imp["pca_features_in"] = list(pca.feature_names_in_)
        feature_imp["pca_components"] = pca.components_
    return dropped, feature_imp


def format_config(config):
    """Format config for saving as a csv"""
    config = pd.json_normalize(config, sep="_")
    config = config.rename(columns={"dataset_dataset_path": "dataset"})
    config = config.drop({"name", "dataset_first_iqm"}, axis=1)
    if config.loc[0, "experiment_classification_threshold"] is None:
        config = config.drop({"experiment_classification_threshold"}, axis=1)
    config.loc[0, "dataset"] = os.path.basename(config.loc[0, "dataset"])
    return config.loc[0].to_dict()


def save_inner_scores(exp, config_, nested_score):
    """This function works on the score of the inner estimator.
    We get a score for each estimator and average it based on the
    performance on the *inner* cross validation loop
    """
    out_path = "scores_inner.csv"
    out_path_agg = "scores_inner_agg.csv"
    out_path_agg_2 = "scores_inner_agg_more.csv"
    params = list(nested_score["estimator"][0].param_grid.keys())
    scores = list(nested_score["estimator"][0].scoring.keys())
    scores = [f"{m}_test_{k}" for k in scores for m in ["mean", "std"]]
    config = format_config(copy.deepcopy(config_))
    config_keys = list(config.keys())
    cols = config_keys + ["outer_split", "inner_idx"] + params + scores
    results = pd.DataFrame(columns=cols)
    for i, gridsearch in enumerate(nested_score["estimator"]):
        len_grid = len(gridsearch.cv_results_["params"])

        for j in range(len_grid):
            cv_step = {k: v for k, v in config.items()}
            cv_step.update(
                {
                    k: v[j]
                    for k, v in gridsearch.cv_results_.items()
                    if k in scores
                }
            )
            cv_step.update(
                {
                    k: str(v)
                    for k, v in gridsearch.cv_results_["params"][j].items()
                }
            )
            cv_step.update({"outer_split": i + 1, "inner_idx": j + 1})
            results = pd.concat(
                [results, pd.DataFrame([cv_step])], ignore_index=True
            )
    results.to_csv(out_path, index=False)
    results.groupby(["outer_split"] + params).mean(numeric_only=True).to_csv(
        out_path_agg, index=False
    )
    results.groupby(params).mean(numeric_only=True).to_csv(
        out_path_agg_2, index=False
    )
    for p in [
        out_path,
        out_path_agg,
        out_path_agg_2,
    ]:
        exp.add_artifact(p)
        os.remove(p)


def save_outer_scores(exp, config_, nested_score, metrics_list):
    """This function extracts and aggregates the results of the outer
    cross validation loop.
    """
    out_path_outer = "scores_outer.csv"
    out_path_outer_2 = "scores_outer_agg.csv"
    params = list(nested_score["estimator"][0].param_grid.keys())
    scores = list(nested_score["estimator"][0].scoring.keys())
    scores = [f"test_{k}" for k in scores]
    config = format_config(copy.deepcopy(config_))
    config_keys = list(config.keys())
    cols = config_keys + ["outer_split"] + params + scores
    n_eval = len(nested_score["estimator"])
    results_best = pd.DataFrame(
        columns=cols + ["features_dropped", "features_importance"]
    )
    for i in range(n_eval):

        cv_step = {k: v for k, v in config.items()}
        est = nested_score["estimator"][0]
        cv_step.update({k: str(v) for k, v in est.best_params_.items()})
        cv_step.update({s: nested_score[s][i] for s in scores})

        (
            cv_step["features_dropped"],
            cv_step["features_importance"],
        ) = get_feature_importance(est.best_estimator_, metrics_list)
        cv_step.update({"outer_split": i + 1})
        results_best = pd.concat(
            [results_best, pd.DataFrame([cv_step])],
            ignore_index=True,
        )
    results_best.to_csv(out_path_outer, index=False)

    ## Aggregating outer cv results into a single line.
    num = results_best.select_dtypes(include="number").columns.tolist()
    str_ = results_best.select_dtypes(include="object").columns.tolist()
    import numpy as np

    def join_col(col):
        col_ = np.array([str(c) for c in col])
        if len(np.unique(col_)) == 1:
            return col_[0]
        else:
            return "; ".join(col_)

    final_num = results_best[num].mean(numeric_only=True)
    final_str = results_best[str_].agg(join_col)
    final = pd.concat([final_num, final_str], axis=0)
    pd.DataFrame(final[results_best.columns]).T.to_csv(
        out_path_outer_2, index=False
    )
    for p in [
        out_path_outer,
        out_path_outer_2,
    ]:
        exp.add_artifact(p)
        os.remove(p)


def save_to_csv(exp, config_, nested_score, metrics_list):
    save_inner_scores(exp, config_, nested_score)
    save_outer_scores(exp, config_, nested_score, metrics_list)
