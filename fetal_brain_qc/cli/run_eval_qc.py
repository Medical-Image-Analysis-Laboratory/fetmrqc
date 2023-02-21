""" QC evaluation: 
Train various models and evaluate them.
"""

import pandas as pd
import numpy as np


def load_dataset(csv_path, first_iqm="centroid"):
    """Load and preprocess metrics, by removing rows containing
    NaNs and splitting the dataframe between `iqms` and the
    `rest` of the information (ratings, subject, etc.)
    """
    if csv_path.endswith("csv"):
        metrics = pd.read_csv(csv_path)
    elif csv_path.endswith("tsv"):
        metrics = pd.read_csv(csv_path, index_col=None, delimiter=r"\s+")
    else:
        raise RuntimeError(
            "Unknown extension for {args.metrics_csv}. Please provide a .tsv or .csv file."
        )
    # Dropping NaN rows
    if any(metrics.isnull()):
        print("NaN entries fround:")
        df = metrics[metrics.isnull().any(axis=1)]
        print(df)
        print(f"Dropping {df.shape[0]} rows")
        metrics = metrics.dropna(axis=0)
    iqm_idx = np.where(df.columns == first_iqm)[0][0]

    # Casting IQMs as float
    cols = metrics.columns[iqm_idx:]
    iqms = metrics[cols]
    types = {col: float if "nan" not in col else bool for col in cols}
    iqms = iqms.astype(types).astype(float)
    rest = metrics[metrics.columns[:iqm_idx]]
    return iqms, rest


def drop_largest_correlation(iqms, threshold=0.95, verbose=True):
    """Given a dataframe of iqms, finds highly correlated features
    and drops them.
    """
    corrmat = iqms.corr()
    # Select upper triangle of correlation matrix
    upper = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [
        column
        for column in upper.columns
        if any(abs(upper[column]) > threshold)
    ]

    if verbose:
        for col in sorted(to_drop):
            df_col = upper[col]
            corr_expl = f"{col} correlated with:\n\t"
            for k, v in df_col[abs(df_col) > threshold].items():
                corr_expl += f"{k} ({v:.2f}), "
            print(corr_expl[:-2])

    print(f"Dropping {len(to_drop)} redundant metrics: {to_drop}\n")
    iqms = iqms.drop(to_drop, axis=1)
    return iqms, to_drop


def z_score(x):
    """Standardize a vector x"""
    return (x - x.mean()) / (x.std() + 1e-8)


def preprocess_iqms(iqms, drop_correlated=True, normalize_features=True):
    """ """
    # Re-order columns: features first, whether there is a nan second
    cols_nan = [col for col in iqms.columns if "_nan" in col]
    cols_rest = [col for col in iqms.columns if "_nan" not in col]
    iqms = iqms[cols_rest + cols_nan]

    # Drop constant columns

    cst_cols = iqms.columns[(iqms.var() == 0).values].tolist()
    print(f"Dropped {len(cst_cols)} constant columns: {cst_cols}\n")
    iqms = iqms.drop(cst_cols, axis=1)

    # Drop highly correlated columns
    if drop_correlated:
        iqms, corr_cols = drop_largest_correlation(
            iqms, threshold=0.92, verbose=False
        )

    if normalize_features:
        iqms = iqms.apply(z_score)
    return iqms, cst_cols, corr_cols


def plot_confusion_matrix(tp, fp, fn, tn):
    """Utility to plot a confusion matrix with precision,
    recall, PPV, NPV
    """
    print("Pred |  Actual val.  | Tot.")
    print("     |  Pos.   Neg.  |")
    print("-----------------------------")
    print(
        f"Pos. | {tp:5.1f}  {fp:5.1f}  | {tp+fp:5.1f} - PPV={tp/(tp+fp):.3f}"
    )
    print(
        f"Neg. | {fn:5.1f}  {tn:5.1f}  | {fn+tn:5.1f} - NPV={tn/(tn+fn):.3f} "
    )
    print(f"----------------------------")
    print(f"Tot. | {tp+fn:5.1f}  {fp+tn:5.1f}  | {tp+fp+fn+tn:5.1f}")
    print(f"Prec={tp/(tp+fn):.3f} - Rec={tn/(fp+tn):.3f}")


from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import LinearSVR
from sklearn.linear_model import (
    # LinearRegression,
    # ElasticNet,
    Ridge,
    Lasso,
    HuberRegressor,
    QuantileRegressor,
)

REGRESSION_MODELS = [
    LinearSVR(max_iter=10000),
    # LinearRegression(),
    # ElasticNet(),
    HuberRegressor(),
    QuantileRegressor(),
    Ridge(),
    Lasso(),
    # tree.DecisionTreeRegressor(),
    GradientBoostingRegressor(),
    GradientBoostingRegressor(loss="absolute_error"),
    GradientBoostingRegressor(loss="huber"),
    AdaBoostRegressor(),
    HistGradientBoostingRegressor(),
    HistGradientBoostingRegressor(loss="absolute_error"),
    RandomForestRegressor(),
    RandomForestRegressor(criterion="absolute_error"),
]


from scipy.stats import spearmanr
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
)
from sklearn.metrics import make_scorer


def spearman_correlation(y_true, y_pred):
    """Compute Spearman correlation between a reference and a prediction"""
    check_consistent_length(y_true, y_pred)

    sp = spearmanr(y_true, y_pred)

    return sp.correlation


REGRESSION_SCORING = {
    "r2": "r2",
    "neg_mae": "neg_mean_absolute_error",
    "neg_median_ae": "neg_median_absolute_error",
    "spearman": make_scorer(spearman_correlation),
}


from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
)

CLASSIFICATION_MODELS = [
    # LogisticRegression(),
    RidgeClassifier(),
    LinearSVC(max_iter=100000),
    DecisionTreeClassifier(),
    # ElasticNet(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
]


def tp(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 1]


def fp(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 1]


def fn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1, 0]


def tn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0]


CLASSIFICATION_SCORING = {
    "acc": "accuracy",
    "prec": "precision",
    "rec": "recall",
    "f1": "f1",
    "tp": make_scorer(tp),
    "fp": make_scorer(fp),
    "fn": make_scorer(fn),
    "tn": make_scorer(tn),
    "roc_auc": "roc_auc",
}


def main():
    import os
    import numpy as np
    import argparse
    from fetal_brain_utils import print_title
    import json
    from sklearn.model_selection import (
        KFold,
        GroupKFold,
        GroupShuffleSplit,
    )
    from sklearn.model_selection import cross_validate

    p = argparse.ArgumentParser(
        description=(
            "Train and evaluate models to predict quality from image-extracted metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--out_folder",
        help="Path where the results will be stored.",
        required=True,
    )

    p.add_argument(
        "--metrics_csv",
        help="Path where the evaluated metrics with ratings are stored.",
        required=True,
    )

    p.add_argument(
        "--regression",
        help="Perform regression on the continuous ratings.",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--classification",
        help="Perform classification on the ratings binarized at `threshold`.",
        dest="regression",
        action="store_false",
    )

    p.add_argument(
        "--threshold",
        help="Threshold for binarization in the classification case.",
        default=1.0,
        type=float,
    )

    p.add_argument(
        "--splits_cv",
        help="Number of splits to be used in the cross-validation",
        default=5,
        type=int,
    )

    p.add_argument(
        "--use_groups_cv",
        help="Whether data should be grouped according to the `group_by_cv` field",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--group_by_cv",
        help="Column to be used to group the data when `use_groups_cv=True`.",
        type=str,
        default="sub_ses",
    )

    p.add_argument(
        "--first_iqm",
        help="Name of the first IQM column in the `metrics_csv` file",
        type=str,
        default="centroid",
    )
    args = p.parse_args()
    print_title("Evaluating QC models")

    os.makedirs(args.out_folder)

    regression = args.regression
    threshold = args.threshold

    iqms, rest = load_dataset(args.metrics_csv, args.first_iqm)

    y = np.array(rest["rating"])
    if args.use_groups_cv:
        assert (
            args.group_by_cv in rest.columns
        ), f"ERROR: {args.group_by_cv} not found in the metrics dataframe."
        group = rest[args.group_by_cv]
    else:
        group = None

    iqms, cst_cols, corr_cols = preprocess_iqms(
        iqms, drop_correlated=True, normalize_features=True
    )
    X = np.array(iqms)

    if args.use_groups_cv:
        kf = GroupShuffleSplit(n_splits=args.splits_cv, random_state=0)
    else:
        kf = KFold(n_splits=args.splits_cv)

    regr = "regression" if regression else "classification"
    print_title(
        f"{regr} - Cross-validation using {kf} with n_splits={args.splits_cv}"
    )
    m_dict = {}

    if regression:
        models = REGRESSION_MODELS
        scoring = REGRESSION_SCORING
    else:
        models = CLASSIFICATION_MODELS
        scoring = CLASSIFICATION_SCORING
        y = [0 if ys < threshold else 1 for ys in y]
    for m in models:
        print(f"Fitting {m} ...", end="")
        out = cross_validate(
            m,
            X=X,
            y=y,
            groups=group,
            cv=kf,
            scoring=scoring,
            # return_estimator=True,
        )
        print(out)
        out.pop("fit_time")
        out.pop("score_time")
        m_dict[m] = out
        print(" done!")

    for m in scoring.keys():
        print(f"\nSorting metrics according to {m} (higher is better)")
        metric = [(clf, m_dict[clf][f"test_{m}"].mean()) for clf in models]
        metric = sorted(
            metric,
            key=lambda x: x[1] if not np.isnan(x[1]) else -1e15,
            reverse=True,
        )
        for clf, v in metric:
            print(f"\t{clf} ({m}={v:.3f})")

    if not regression:
        print_title(
            f"Considering threshold={threshold} - averaged confusion matrix.",
            char="#",
        )
        for clf in models:
            tp = m_dict[clf]["test_tp"].mean()
            fp = m_dict[clf]["test_fp"].mean()
            fn = m_dict[clf]["test_fn"].mean()
            tn = m_dict[clf]["test_tn"].mean()
            print_title(str(clf), center=False)
            plot_confusion_matrix(tp, fp, fn, tn)

    result_df = pd.DataFrame(m_dict).T
    result_df["split"] = [np.arange(1, args.splits_cv + 1)] * result_df.shape[
        0
    ]
    result_df = result_df.explode(list(result_df.columns))
    result_df = result_df.reset_index().set_index(["index", "split"])

    result_df.to_csv(os.path.join(args.out_folder, "results.csv"))

    out_dict = args.__dict__
    out_dict["iqms"] = iqms.columns.tolist()
    out_dict["drop_cst_cols"] = cst_cols
    out_dict["drop_corr_cols"] = corr_cols
    with open(os.path.join(args.out_folder, "results.json"), "w") as f:
        json.dump(out_dict, f, indent=4)
    return 0


if __name__ == "__main__":
    main()
