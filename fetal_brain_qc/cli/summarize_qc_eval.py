""" QC summary
"""


def main():
    import os
    import numpy as np
    import pandas as pd
    import argparse
    from fetal_brain_utils import print_title
    import json

    p = argparse.ArgumentParser(
        description=("Summarize the evaluation"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--eval_folder",
        help="Base folder where the results of the evaluation are stored",
        default=None,
    )

    args = p.parse_args()
    files = os.listdir(args.eval_folder)
    files = [
        os.path.join(args.eval_folder, f.split(".")[0])
        for f in files
        if f.endswith(".csv")
    ]
    print_list = []
    for f in files:
        df = pd.read_csv(f + ".csv")
        with open(f + ".json", "r") as jf:
            metadata = json.load(jf)
        df = df.drop("split", axis=1)
        if "classification" in f:
            df = df.drop(["test_fp", "test_tn", "test_tp", "test_fn"], axis=1)
        format = lambda x: f"{x:3f}" if isinstance(x, float) else f"{x}"
        avg = df.groupby("index").mean()
        avg_str = avg.apply(lambda series: series.apply(lambda x: f"{x:5.3f}"))

        std = (
            df.groupby("index")
            .std()
            .apply(lambda series: series.apply(lambda x: f"{x:5.3f}"))
        )
        avg_str = avg_str + "+-" + std
        for col in avg:
            val = avg_str.loc[avg[col].idxmax(), col]
            avg_str.loc[avg[col].idxmax(), col] = f"**{val}**"
        regr = "Regression" if metadata["regression"] else "Classification"
        nsplits = metadata["splits_cv"]

        meta_str = f"Table. {regr}: {nsplits} splits"
        if metadata["use_groups_cv"]:
            group = metadata["group_by_cv"]
            meta_str += f", grouping: {group}"
        if not metadata["regression"]:
            threshold = metadata["threshold"]
            meta_str += f", threshold = {threshold}"
        if (
            "drop_correlated" in metadata.keys()
            and not metadata["drop_correlated"]
        ):
            meta_str += ", keeping correlated features"
        if (
            "norm_features" in metadata.keys()
            and not metadata["norm_features"]
        ):
            meta_str += ", unnormalized features"
        print_title(meta_str)
        print(avg_str.to_markdown())
        # print(avg, std, avg + "+-" + std)

    return 0


if __name__ == "__main__":
    main()
