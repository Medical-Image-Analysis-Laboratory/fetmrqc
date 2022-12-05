""" Image quality assessment based on fetal-IQA. 

Based on the code from Junshen Xu at
https://github.com/daviddmc/fetal-IQA
"""

# Import libraries


def main():
    import os
    import torch
    import argparse
    import pandas as pd
    from pathlib import Path
    from fetal_brain_qc.utils import csv_to_list
    from fetal_brain_qc.fetal_IQA import eval_model, resnet34
    from fetal_brain_qc.definitions import (
        FETAL_IQA_CKPT,
        FETAL_STACK_IQA_CKPT,
    )
    from fetal_brain_qc.metrics import DEFAULT_METRICS, LRStackMetrics

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(
        description=(
            "Performs deep learning-based, slice-wise fetal brain"
            " image quality assessment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--out-csv",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    parser.add_argument(
        "--metrics",
        help="Metrics to be evaluated.",
        default=DEFAULT_METRICS,
        nargs="+",
    )

    parser.add_argument(
        "--bids-csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )
    parser.add_argument(
        "--ckpt_path_slice_iqa",
        help="Path to the checkpoint of the fetal IQA pytorch model (by Junshen Xu at MIT).",
        default=FETAL_IQA_CKPT,
    )

    parser.add_argument(
        "--ckpt_path_stack_iqa",
        help="Path to the checkpoint of the fetal IQA tensorflow model (by Ivan Legorreta FNNDSC).",
        default=FETAL_STACK_IQA_CKPT,
    )
    parser.add_argument(
        "--device",
        help="Device to be used for the deep learning model.",
        default="cuda:0",
    )

    args = parser.parse_args()
    bids_list = csv_to_list(args.bids_csv)
    print("Running ")
    lr_metrics = LRStackMetrics(
        args.metrics,
        ckpt_stack_iqa=args.ckpt_path_stack_iqa,
        ckpt_slice_iqa=args.ckpt_path_slice_iqa,
        device=args.device,
    )
    metrics_dict = {}
    for run in bids_list:
        # Loading data
        name = Path(run["im"]).name
        print(f"Processing subject {name}")
        metrics_dict[run["name"]] = lr_metrics.evaluate_metrics(
            run["im"], run["mask"]
        )
    df_base = pd.DataFrame.from_dict(bids_list)
    df_base = df_base.set_index("name")
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    df = pd.concat([df_base, df], axis=1, join="inner")
    df.to_csv(args.out_csv)
    return 0


if __name__ == "__main__":
    main()
