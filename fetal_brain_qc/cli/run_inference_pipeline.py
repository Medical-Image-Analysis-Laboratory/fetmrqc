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
"""
FetMRQC inference pipeline script, calling successively the following steps:
1. qc_brain_extraction 2. qc_list_bids_csv 3. qc_segmentation 4. qc_compute_iqms 5. qc_run_inference
Given a BIDS dataset, compute brain masks, segmentations and FetMRQC IQMs.
Then computes inference using a pretrained FetMRQC model
"""
import argparse
import os
from fetal_brain_qc.definitions import MASK_PATTERN, BRAIN_CKPT
from fetal_brain_qc.qc_evaluation import METRICS, METRICS_SEG
from fetal_brain_qc.definitions import FETMRQC20, FETMRQC20_METRICS
import json
from fetal_brain_qc.cli.build_run_parsers import build_inference_parser

IQMS_NO_NAN = [iqm for iqm in METRICS + METRICS_SEG if "_nan" not in iqm]


def run_cmd(cmd):
    flag = os.system(cmd)
    if flag != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory, computes brain masks and segmentations,"
            " uses them to extract IQMs and perform inference using one of "
            "the pretrained FetMRQC models. The output is a CSV file containing"
            " the predictions and IQMs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_inference_parser(parser)

    args = parser.parse_args()

    # Running brain extraction
    cmd = (
        "qc_brain_extraction "
        f"--bids_dir {args.bids_dir}  "
        f"--masks_dir {args.masks_dir} "
        f"--mask_pattern {args.mask_pattern} "
        f"--ckpt_path {args.ckpt_path}"
    )
    run_cmd(cmd)

    # Running list of BIDS csv file
    cmd = (
        "qc_list_bids_csv "
        f"--bids_dir {args.bids_dir} "
        f"--mask_patterns_base {args.masks_dir} "
        f"--mask_patterns {args.mask_pattern} "
        f"--out_csv {args.bids_csv} "
        f"--seed {args.seed}"
    )
    run_cmd(cmd)

    # Running segmentation
    cmd = (
        "qc_segmentation "
        f"--bids_csv {args.bids_csv} "
        f"--out_path {args.seg_dir} "
        f"--device {args.device} "
    )
    run_cmd(cmd)
    # Running IQMs computation

    cmd_iqms = "" if args.fetmrqc20_iqms else "--use_all_metrics"
    cmd = (
        "qc_compute_iqms "
        f"--bids_csv {args.bids_csv} "
        f"--out_csv {args.iqms_csv} "
        f"{cmd_iqms} "
        "--verbose "
        f"--device {args.device} "
    )
    run_cmd(cmd)

    # Running inference
    fetmrqc20 = "--fetmrqc20 " if args.fetmrqc20_iqms else ""
    cmd = (
        "qc_inference "
        f"--iqms_csv {args.iqms_csv} "
        f"--out_csv {args.out_csv} "
        f"--regression --classification "
        f"{fetmrqc20}"
    )
    print(cmd)
    run_cmd(cmd)


if __name__ == "__main__":
    main()
