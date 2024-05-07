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
from fetal_brain_qc.cli.build_run_parsers import build_reports_parser


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
    build_reports_parser(parser)
    args = parser.parse_args()
    reports_dir = (
        os.path.join(args.out_dir, "derivatives/reports")
        if args.reports_dir is None
        else args.reports_dir
    )
    os.makedirs(reports_dir, exist_ok=True)
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
        f"--out_dir {reports_dir} "
    )

    os.system(cmd)

    # Running the index generation
    cmd = (
        "qc_generate_index "
        f"--reports_dirs {reports_dir} "
        f"--seed {args.seed}"
    )
    os.system(cmd)


if __name__ == "__main__":
    main()
