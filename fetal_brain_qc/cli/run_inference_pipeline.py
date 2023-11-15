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
from fetal_brain_qc.definitions import FETMRQC20
import json

IQMS_NO_NAN = [iqm for iqm in METRICS + METRICS_SEG if "_nan" not in iqm]


def run_cmd(cmd):
    flag = os.system(cmd)
    if flag != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def build_parser(parser):
    parser.add_argument(
        "--bids_dir",
        required=True,
        help="BIDS directory containing the LR series.",
    )

    parser.add_argument(
        "--masks_dir",
        help=(
            "Root of the BIDS directory where brain masks will be/are stored. "
            "If masks already exist, they will be used."
        ),
        required=True,
    )

    parser.add_argument(
        "--seg_dir",
        help=(
            "Root of the directory where brain segmentations will be stored. "
            "If segmentations already exist, they will be used."
        ),
        required=True,
    )
    parser.add_argument(
        "--bids_csv",
        help="CSV file where the list of available LR series and masks will be stored.",
        default="bids_csv.csv",
    )
    parser.add_argument(
        "--iqms_csv",
        help="CSV file where the computed IQMs will be stored.",
        default="iqms_csv.csv",
    )

    parser.add_argument(
        "--out_csv",
        help="CSV file where the predictions from FetMRQC will be stored.",
        default="out_csv.csv",
    )

    parser.add_argument(
        "--iqms",
        help="List of IQMs that will be computed",
        nargs="+",
        default="all",
    )

    parser.add_argument(
        "--fetmrqc20_iqms",
        help="Whether the IQMs from FetMRQC-20 should be computed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--use_all_iqms",
        help="Whether all IQMs should be computed",
        default=False,
        action="store_false",
        dest="fetmrqc20_iqms",
    )
    parser.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the MONAIfbs model.",
        default=BRAIN_CKPT,
    )

    parser.add_argument(
        "--mask_pattern",
        help=(
            "Pattern according to which the masks will be stored.\n "
            'By default, masks will be stored in "<masks_dir>/sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}'
            '[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz", and the different fields will be '
            "substituted based on the structure of bids_dir."
        ),
        type=str,
        default=MASK_PATTERN,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to control the randomization (to be used with randomize=True).",
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
        "--custom_model",
        help="Path to a custom model, trained using run_train_fetmrqc.py.",
        default=None,
        type=str,
    )


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
    build_parser(parser)

    args = parser.parse_args()

    iqms = IQMS_NO_NAN if args.iqms == "all" else args.iqms
    iqms = [iqm for iqm in iqms if "_nan" not in iqm]
    if args.fetmrqc20_iqms:
        assert (
            iqms == IQMS_NO_NAN
        ), "Cannot take a custom set of IQMs when using FetMRQC-20 (--fetmrqc20_iqms)"

    if iqms != IQMS_NO_NAN:
        for iqm in iqms:
            assert iqm in IQMS_NO_NAN, f"The IQM {iqm} is not available."

        assert (
            args.custom_model is not None
        ), "When extracting a custom set of IQMs, a custom model must be provided (--custom_model)."

    if args.custom_model is not None:
        custom_json = args.custom_model.replace(".joblib", ".json")
        with open(custom_json, "r") as f:
            json_dict = json.load(f)
        for iqm in json_dict["iqms"]:
            assert (
                iqm in iqms
            ), f"The IQM {iqm} from the custom model is not in the provided IQMs (--iqms)."
        assert (
            args.classification == json_dict["classification"]
        ), "Mismatch between the task of the custom model and the provided mode (--classification/--regression)."

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
    )
    run_cmd(cmd)
    # Running IQMs computation
    iqms = FETMRQC20 if args.fetmrqc20_iqms else iqms

    cmd = (
        "qc_compute_iqms "
        f"--bids_csv {args.bids_csv} "
        f"--out_csv {args.iqms_csv} "
        f"--metrics {' '.join(iqms)}"
    )
    run_cmd(cmd)

    # Running inference
    task = "--classification " if args.classification else "--regression "
    custom_model = (
        f"--custom_model {args.custom_model}" if args.custom_model else ""
    )
    fetmrqc20 = "--fetmrqc20 " if args.fetmrqc20_iqms else ""
    cmd = (
        "qc_inference "
        f"--iqms_csv {args.iqms_csv} "
        f"--out_csv {args.out_csv} "
        f"{task}"
        f"{custom_model} "
        f"{fetmrqc20}"
    )
    print(cmd)
    run_cmd(cmd)


if __name__ == "__main__":
    main()
