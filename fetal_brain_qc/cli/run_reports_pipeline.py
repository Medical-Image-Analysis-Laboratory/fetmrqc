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
"""Reports generation pipeline script, calling successively the following steps:
1. qc_brain_extraction 2. qc_list_bids_csv 3. qc_generate_reports 4. qc_generate_index
Given a BIDS dataset, generates a report for each subject/session/run.
"""

import argparse
import os
from fetal_brain_qc.definitions import MASK_PATTERN, BRAIN_CKPT


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory, computes the brain masks using MONAIfbs"
            " and uses the masks to compute visual reports that can be"
            " used for manual rating."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bids_dir",
        required=True,
        help="BIDS directory containing the LR series.",
    )

    parser.add_argument(
        "--masks_dir",
        help="Root of the BIDS directory where brain masks will be stored.",
        required=True,
    )

    parser.add_argument(
        "--reports_dir",
        help="Directory where the reports will be stored.",
        required=True,
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
        "--bids_csv",
        help="CSV file where the list of available LR series and masks will be stored.",
        default="bids_csv.csv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
    args = parser.parse_args()

    # Running brain extraction
    cmd = (
        "qc_brain_extraction "
        f"--bids_dir {args.bids_dir}  "
        f"--masks_dir {args.masks_dir} "
        f"--mask_pattern {args.mask_pattern} "
        f"--ckpt_path {args.ckpt_path}"
    )
    os.system(cmd)

    # Running list of BIDS csv file
    cmd = (
        "qc_list_bids_csv "
        f"--bids_dir {args.bids_dir} "
        f"--mask_patterns_base {args.masks_dir} "
        f"--mask_patterns {args.mask_pattern} "
        f"--out_csv {args.bids_csv} "
        f"--seed {args.seed}"
    )
    os.system(cmd)

    # Running the reports generation
    cmd = (
        "qc_generate_reports "
        f"--bids_csv {args.bids_csv} "
        f"--out_dir {args.reports_dir} "
    )

    os.system(cmd)

    # Running the index generation
    cmd = (
        "qc_generate_index "
        f"--reports_dirs {args.reports_dir} "
        f"--seed {args.seed}"
    )
    os.system(cmd)


if __name__ == "__main__":
    main()
