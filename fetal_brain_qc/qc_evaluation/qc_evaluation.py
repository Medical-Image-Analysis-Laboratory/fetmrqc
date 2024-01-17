# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
import numpy as np
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
    ElasticNet,
    Ridge,
    Lasso,
    HuberRegressor,
    QuantileRegressor,
    LogisticRegression,
    RidgeClassifier,
)
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from sklearn.utils.validation import (
    check_consistent_length,
)
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from functools import wraps
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from functools import partial


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
    corr_cols = []
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
    print("----------------------------")
    print(f"Tot. | {tp+fn:5.1f}  {fp+tn:5.1f}  | {tp+fp+fn+tn:5.1f}")
    print(f"Prec={tp/(tp+fn):.3f} - Rec={tn/(fp+tn):.3f}")


##
# Defining scores for the evaluation
##


def spearman_correlation(y_true, y_pred):
    """Compute Spearman correlation between a reference and a prediction"""
    check_consistent_length(y_true, y_pred)

    sp = spearmanr(y_true, y_pred)
    # print(f"Spearman: input shape {y_true.shape} - result: {sp.correlation}")
    return sp.correlation


def tp(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 1:
        return sum(y_true)
    return cm[1, 1]


def fp(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 1:
        return 0
    return cm[0, 1]


def fn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 1:
        return 0
    return cm[1, 0]


def tn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 1:
        return 1 - sum(y_true)
    return cm[0, 0]


def nan_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return np.nan
    else:
        return roc_auc_score(y_true, y_pred)


##
# Defining default evaluation settings
##

REGRESSION_MODELS = [
    LinearSVR(),
    LinearRegression(),
    ElasticNet(),
    HuberRegressor(),
    QuantileRegressor(),
    Ridge(),
    Lasso(),
    GradientBoostingRegressor(),
    AdaBoostRegressor(),
    HistGradientBoostingRegressor(),
    RandomForestRegressor(),
    XGBRegressor(),
]


### BINARIZING

THRESHOLD = 1.0


def binarize_metric_input(metric, threshold=THRESHOLD):
    @wraps(metric)
    def binarized_metric(y_true, y_pred):
        y_true = y_true > threshold
        y_pred = y_pred > threshold
        if len(np.unique(y_true)) == 1:
            return np.nan
        return metric(y_true, y_pred)

    return binarized_metric


def binarized_roc(y_true, y_pred):
    y_true = y_true > THRESHOLD
    pred_proba = y_pred / 4.0  # Put in range [0,1]
    if len(np.unique(y_true)) == 1:
        return np.nan
    return roc_auc_score(y_true, pred_proba)


acc = binarize_metric_input(accuracy_score)
prec = binarize_metric_input(precision_score)
rec = binarize_metric_input(recall_score)
f1 = binarize_metric_input(f1_score)
f1w = binarize_metric_input(partial(f1_score, average="weighted"))
# roc_auc = binarized_roc  # binarize_metric_input(nan_roc_auc)
tp_ = binarize_metric_input(tp)
fp_ = binarize_metric_input(fp)
fn_ = binarize_metric_input(fn)
tn_ = binarize_metric_input(tn)

REGRESSION_SCORING = {
    "r2": "r2",
    "neg_mae": "neg_mean_absolute_error",
    "neg_median_ae": "neg_median_absolute_error",
    "spearman": make_scorer(spearman_correlation),
    "acc": make_scorer(acc),
    "prec": make_scorer(prec),
    "rec": make_scorer(rec),
    "f1": make_scorer(f1),
    "f1w": make_scorer(f1),
    "roc_auc": make_scorer(binarized_roc),
    "tp": make_scorer(tp_),
    "fp": make_scorer(fp_),
    "fn": make_scorer(fn_),
    "tn": make_scorer(tn_),
}


CLASSIFICATION_MODELS = [
    LogisticRegression(),
    RidgeClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
]

CLASSIFICATION_SCORING = {
    "acc": "accuracy",
    "bacc": "balanced_accuracy",
    "prec": "precision",
    "rec": "recall",
    "f1": "f1",
    "f1w": "f1_weighted",
    "tp": make_scorer(tp),
    "fp": make_scorer(fp),
    "fn": make_scorer(fn),
    "tn": make_scorer(tn),
    "roc_auc": make_scorer(
        nan_roc_auc, greater_is_better=True, needs_threshold=True
    ),
    "roc_auc_old": make_scorer(nan_roc_auc),
}
