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
        FETAL_IQA_CKPT,
        FETAL_FETMRQC_CLF_CKPT,
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
        "--task",
        help="Whether a discrete score (regression) or a binary score (classification) is returned.",
        default="classification",
        choices=["classification", "regression"],
    )

    args = parser.parse_args()
    bids_df = pd.read_csv(args.bids_csv)

    task = args.task
    ckpt_path = args.ckpt_path

    if ckpt_path is None:
        if task == "classification":
            ckpt_path = FETAL_FETMRQC_CLF_CKPT
        elif task == "regression":
            ckpt_path = FETAL_IQA_CKPT

    model = joblib.load(ckpt_path)

    print_title(f"Running FetMRQC inference ({task}).")
    out_path = Path(args.out_csv)
    os.makedirs(out_path.parent, exist_ok=True)

    test_x = bids_df[METRICS + METRICS_SEG]
    test_y = model.predict(test_x)
    import pdb

    im_loc = bids_df.columns.get_loc("im")
    bids_df.insert(im_loc - 1, "fetmrqc_qc_passed", test_y.astype(int))
    bids_df.to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    main()
