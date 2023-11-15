# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
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
""" Command line interface for the extraction of IQMs from fetal brain images
using FetMRQC
"""


def main(argv=None):
    import os
    import numpy as np
    import torch
    import argparse
    import pandas as pd
    from pathlib import Path
    from fetal_brain_utils import csv_to_list, print_title
    from fetal_brain_qc.fetal_IQA import eval_model, resnet34
    from fetal_brain_qc.definitions import (
        FETAL_IQA_CKPT,
        FETAL_STACK_IQA_CKPT,
    )
    from fetal_brain_qc.metrics import DEFAULT_METRICS, LRStackMetrics

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    p = argparse.ArgumentParser(
        description=("Computes quality metrics from given images."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--out_csv",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    p.add_argument(
        "--metrics",
        help="Metrics to be evaluated.",
        default=DEFAULT_METRICS,
        nargs="+",
    )

    p.add_argument(
        "--use_all_metrics",
        help="Whether all metrics should be evaluated",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument(
        "--normalization",
        help="Whether input data should be normalized",
        default=None,
        choices=[None, "sub_ses", "site", "run"],
    )

    p.add_argument(
        "--bids_csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )
    p.add_argument(
        "--ckpt_path_slice_iqa",
        help="Path to the checkpoint of the fetal IQA pytorch model (by Junshen Xu at MIT).",
        default=FETAL_IQA_CKPT,
    )

    p.add_argument(
        "--ckpt_path_stack_iqa",
        help="Path to the checkpoint of the fetal IQA tensorflow model (by Ivan Legorreta FNNDSC).",
        default=FETAL_STACK_IQA_CKPT,
    )
    p.add_argument(
        "--device",
        help="Device to be used for the deep learning model.",
        default="cuda:0",
    )

    p.add_argument(
        "--continue_run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether QC run should re-use existing results if a metrics.csv file at "
            "`out_path`/metrics.csv."
        ),
    )

    p.add_argument(
        "--use_prob_seg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to use the probability segmentation or the binary segmentation."
        ),
    )

    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable verbose."),
    )

    args = p.parse_args(argv)
    bids_list = csv_to_list(args.bids_csv)
    print_title("Computing IQMs")

    lr_metrics = LRStackMetrics(
        ckpt_stack_iqa=args.ckpt_path_stack_iqa,
        ckpt_slice_iqa=args.ckpt_path_slice_iqa,
        device=args.device,
        verbose=args.verbose,
    )

    if args.use_all_metrics:
        if args.metrics != DEFAULT_METRICS:
            print(
                f"WARNING: --use_all_metrics is enabled. Ignoring custom metrics {args.metrics}"
            )
        lr_metrics.set_metrics(lr_metrics.get_all_metrics())
    else:
        lr_metrics.set_metrics(args.metrics)

    lr_metrics.normalize_dataset(bids_list, args.normalization)

    metrics_dict = {}
    df_base = pd.DataFrame.from_dict(bids_list)
    df_base = df_base.set_index("name")
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    # If a file is found, continue.
    if os.path.isfile(args.out_csv) and args.continue_run:
        print("\tCONTINUING FROM A PREVIOUSLY FOUND RUN.")
        df = pd.read_csv(args.out_csv).set_index("name")
        metrics_dict = df.to_dict(orient="index")
        # Remove duplicate keys
        metrics_dict = {
            k: {k2: v2 for k2, v2 in v.items() if k2 not in df_base.columns}
            for k, v in metrics_dict.items()
        }
    if args.use_prob_seg:
        seg_key = "seg_proba"
    else:
        seg_key = "seg"
    for run in bids_list:
        # Loading data
        name = Path(run["im"]).name
        if run["name"] in metrics_dict.keys():
            print(f"Subject {name} found in {args.out_csv}.")
            continue

        print(f"Processing subject {name}")
        import warnings
        import sys

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stdo = sys.stdout  # backup current stdout
            sys.stdout = open(os.devnull, "w")

            metrics_dict[run["name"]] = lr_metrics.evaluate_metrics(
                run["im"], run["mask"], run[seg_key]
            )
            sys.stdout = stdo
        df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        df = pd.concat([df_base, df], axis=1, join="inner")
        df.index.name = "name"
        df.to_csv(args.out_csv)
    return 0


if __name__ == "__main__":
    main()
