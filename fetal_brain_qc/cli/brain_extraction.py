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


def main():
    import argparse
    import os
    from bids import BIDSLayout
    from fetal_brain_qc.definitions import MASK_PATTERN, BRAIN_CKPT
    from fetal_brain_qc.brain_extraction import (
        bidsify_monaifbs,
        run_brain_extraction,
        mask_was_found,
    )
    from fetal_brain_utils import iter_bids, print_title
    from fetal_brain_qc.utils import fill_pattern
    from pathlib import Path

    p = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory and computes the brain masks using MONAIfbs "
            " (https://github.com/gift-surg/MONAIfbs/tree/main). Save the masks"
            " into the `masks_dir` folder, follwing the same hierarchy as the `bids_dir`"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--bids_dir",
        help="BIDS directory containing the LR series.",
        required=True,
    )

    p.add_argument(
        "--masks_dir",
        help="Root of the BIDS directory where brain masks will be stored.",
        required=True,
    )

    p.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the MONAIfbs model.",
        default=BRAIN_CKPT,
    )

    p.add_argument(
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

    args = p.parse_args()
    print_title("Running Brain extraction")
    bids_layout = BIDSLayout(args.bids_dir, validate=False)

    # Create the output directory
    os.makedirs(args.masks_dir, exist_ok=True)
    # Creating the pattern to save the resulting masks
    mask_pattern = os.path.join(
        os.path.abspath(args.masks_dir), args.mask_pattern
    )

    # Listing all files if there isn't already a mask for them.
    files_filtered = [
        (sub, ses, run, out)
        for sub, ses, run, out in iter_bids(bids_layout)
        if not mask_was_found(bids_layout, sub, ses, run, mask_pattern)
    ]

    if len(files_filtered) == 0:
        print("All masks were already computed")
    else:
        # Create a tmp directory for the output of monaifbs segmentation
        masks_tmp = os.path.join(args.masks_dir, "tmp")
        files_paths = [o[3] for o in files_filtered]
        run_brain_extraction(files_paths, masks_tmp, brain_ckpt=args.ckpt_path)
        # Move files to their definitive location
        bidsify_monaifbs(files_filtered, bids_layout, mask_pattern, masks_tmp)

    return 0


if __name__ == "__main__":
    main()
