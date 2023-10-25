"""
FetMRQC inference script
"""

# Import libraries


def main():
    import os
    import argparse
    import pandas as pd
    from pathlib import Path
    import joblib
    from fetal_brain_utils import print_title
    from fetal_brain_qc.definitions import (
        FETAL_FETMRQC_REG_CKPT,
        FETAL_FETMRQC_CLF_CKPT,
        FETAL_FETMRQC20_REG_CKPT,
        FETAL_FETMRQC20_CLF_CKPT,
        FETMRQC20,
    )
    from fetal_brain_qc.qc_evaluation import METRICS, METRICS_SEG

    parser = argparse.ArgumentParser(
        description=("Performs FetMRQC inference, given a pretrained model."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bids-csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )

    parser.add_argument(
        "--out-csv",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    parser.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the fetal IQA pytorch model.",
        default=None,
    )

    parser.add_argument(
        "--classification",
        help="Whether to perform classification or regression.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--regression",
        help="Whether to perform classification or regression.",
        dest="classification",
        action="store_false",
    )

    parser.add_argument(
        "--fetmrqc20",
        help="Whether to use FetMRQC20 IQMs.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()
    bids_df = pd.read_csv(args.bids_csv)

    task = "classification" if args.classification else "regression"
    ckpt_path = args.ckpt_path

    if ckpt_path is None:
        if task == "classification":
            if not args.fetmrqc20:
                ckpt_path = FETAL_FETMRQC_CLF_CKPT
            else:
                ckpt_path = FETAL_FETMRQC20_CLF_CKPT
        elif task == "regression":
            if not args.fetmrqc20:
                ckpt_path = FETAL_FETMRQC_REG_CKPT
            else:
                ckpt_path = FETAL_FETMRQC20_REG_CKPT

    model = joblib.load(ckpt_path)

    print_title(f"Running FetMRQC inference ({task}).")
    out_path = Path(args.out_csv)
    os.makedirs(out_path.parent, exist_ok=True)
    if not args.fetmrqc20:
        iqms = METRICS + METRICS_SEG
    else:
        iqms = FETMRQC20
    test_x = bids_df[iqms]
    test_y = model.predict(test_x)

    if task == "classification":
        bids_df.insert(0, "fetmrqc", test_y.astype(int))
    else:
        bids_df.insert(0, "fetmrqc", test_y)

    bids_df.to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    main()
