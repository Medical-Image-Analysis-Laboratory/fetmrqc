""" QC evaluation: 
Train various models and evaluate them.
"""


def main():
    import os
    import numpy as np
    import pandas as pd
    import argparse
    from fetal_brain_utils import print_title
    import json
    from sklearn.model_selection import (
        KFold,
        GroupKFold,
        GroupShuffleSplit,
    )
    from sklearn.model_selection import cross_validate
    from ..qc_evaluation.qc_evaluation import (
        load_dataset,
        preprocess_iqms,
        plot_confusion_matrix,
        REGRESSION_MODELS,
        REGRESSION_SCORING,
        CLASSIFICATION_MODELS,
        CLASSIFICATION_SCORING,
    )

    p = argparse.ArgumentParser(
        description=(
            "Train and evaluate models to predict quality from image-extracted metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--out_path",
        help="Base path where the results will be stored (file name *without* extension).",
        default=None,
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

    p.add_argument(
        "--drop_correlated",
        help="Whether correlated features should be dropped",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--norm_features",
        help="Whether each feature should be normalized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = p.parse_args()
    print_title("Evaluating QC models")
    out_folder = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(out_folder, exist_ok=True)

    regression = args.regression
    threshold = args.threshold
    regr = "regression" if regression else "classification"

    out_base = f"{args.out_path}_nsplits-{args.splits_cv}_norm-{args.norm_features}_dropCorrelated-{args.drop_correlated}"
    if not regression:
        thresh_str = str(threshold).replace(".", "p")
        out_base += f"_threshold-{thresh_str}"
    if args.use_groups_cv:
        out_base += f"_group-{args.group_by_cv}"
    out_base += f"_mode-{regr}"

    assert not os.path.isfile(
        out_base + ".csv"
    ), f"ERROR: {out_base}.csv already exists"
    assert not os.path.isfile(
        out_base + ".json"
    ), f"ERROR: {out_base}.json already exists"

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
        iqms,
        drop_correlated=args.drop_correlated,
        normalize_features=args.norm_features,
    )
    X = np.array(iqms)

    if args.use_groups_cv:
        kf = GroupShuffleSplit(n_splits=args.splits_cv, random_state=0)
    else:
        kf = KFold(n_splits=args.splits_cv)

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

    result_df.to_csv(out_base + ".csv")

    out_dict = args.__dict__
    out_dict["iqms"] = iqms.columns.tolist()
    out_dict["drop_cst_cols"] = cst_cols
    out_dict["drop_corr_cols"] = corr_cols
    with open(out_base + ".json", "w") as f:
        json.dump(out_dict, f, indent=4)
    return 0


if __name__ == "__main__":
    main()
