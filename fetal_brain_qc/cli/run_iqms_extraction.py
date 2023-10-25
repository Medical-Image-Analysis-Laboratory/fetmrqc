""" Command line interface for the extraction of IQMs from fetal brain images
using FetMRQC
"""

# Import libraries
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
import multiprocessing

DL_METRICS = [
    "dl_stack_iqa_full",
    "dl_slice_iqa",
    "dl_slice_iqa_cropped",
    "dl_slice_iqa_full",
    "dl_slice_iqa_full_cropped",
    "dl_slice_iqa_pos_only_full",
]


def setup_lr_metrics(args, use_dl=False, metrics=None):
    """ "Setting up the LR metrics class for computation of the IQMs."""
    if use_dl:
        lr_metrics = LRStackMetrics(
            ckpt_stack_iqa=args.ckpt_path_stack_iqa,
            ckpt_slice_iqa=args.ckpt_path_slice_iqa,
            device=args.device,
            verbose=args.verbose,
        )
    else:
        lr_metrics = LRStackMetrics(verbose=args.verbose)

    if args.use_all_metrics:
        if args.metrics != DEFAULT_METRICS:
            print(
                f"WARNING: --use_all_metrics is enabled. Ignoring custom metrics {args.metrics}"
            )
        lr_metrics.set_metrics(lr_metrics.get_all_metrics())
    else:
        if metrics is None:
            metrics = args.metrics
        lr_metrics.set_metrics(metrics)

    return lr_metrics


def process_subject(args_run):
    """Process a single subject, can be executed in parallel."""
    args, run = args_run
    # Remove DL_METRICS from args.metrics
    metrics = [m for m in args.metrics if m not in DL_METRICS]
    lr_metrics = setup_lr_metrics(args, metrics)
    name = Path(run["im"]).name
    id_ = run["name"]

    print(f"Processing subject {name}")
    seg_key = "seg_proba" if args.use_prob_seg else "seg"
    out = lr_metrics.evaluate_metrics(run["im"], run["mask"], run[seg_key])
    return id_, out


def process_subjects_dl(args, bids_list, results):
    """Process subjects using the deep learning IQMs.
    This is executed sequentially.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    metrics = [m for m in args.metrics if m in DL_METRICS]
    lr_metrics = setup_lr_metrics(args, metrics)
    if len(metrics) == 0:
        return results
    for run in bids_list:
        # Loading data
        name = Path(run["im"]).name
        id_ = run["name"]
        print(f"Processing subject {name}")
        seg_key = "seg_proba" if args.use_prob_seg else "seg"
        out = lr_metrics.evaluate_metrics(run["im"], run["mask"], run[seg_key])
        results[id_].update(out)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(
        description=("Computes quality metrics from given images."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--out-csv",
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
        "--bids-csv",
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
        "--continue-run",
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
        "--nprocs",
        type=int,
        default=4,
        help=("Number of processes to use for multiprocessing."),
    )

    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Enable verbose."),
    )

    args = p.parse_args(argv)
    bids_list = csv_to_list(args.bids_csv)
    print_title("Running QC evaluation")

    df_base = pd.DataFrame.from_dict(bids_list)
    df_base = df_base.set_index("name")
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    results_old = {}
    # If a file is found, continue.
    if os.path.isfile(args.out_csv) and args.continue_run:
        print("\tCONTINUING FROM A PREVIOUSLY FOUND RUN.")
        results_old = (
            pd.read_csv(args.out_csv).set_index("name").to_dict(orient="index")
        )
        # Remove duplicate keys
        results_old = {
            k: {k2: v2 for k2, v2 in v.items() if k2 not in df_base.columns}
            for k, v in results_old.items()
        }

    bids_list = [
        run for run in bids_list if run["name"] not in results_old.keys()
    ]
    # Start processes with data allocation
    pool = multiprocessing.Pool(processes=args.nprocs)
    import time

    start = time.time()
    # Apply the function to the data using the pool
    results = pool.map(
        process_subject, zip([args] * len(bids_list), bids_list)
    )
    results = {res[0]: res[1] for res in results}

    if results_old:
        results.update(results_old)
    pool.close()
    pool.join()

    results = process_subjects_dl(args, bids_list, results)
    # Close the pool and wait for the work to finish
    df = pd.DataFrame.from_dict(results, orient="index")
    df = pd.concat([df_base, df], axis=1, join="inner")
    df.index.name = "name"
    df.to_csv(args.out_csv)
    print(f"Running time {time.time() - start}")

    return 0


if __name__ == "__main__":
    main()
