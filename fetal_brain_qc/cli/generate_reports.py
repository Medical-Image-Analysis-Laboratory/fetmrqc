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
    from fetal_brain_qc.report import generate_report
    from fetal_brain_utils import csv_to_list, print_title

    p = argparse.ArgumentParser(
        "Given a BIDS CSV file, generates visual reports for annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--bids_csv",
        help="Path where the bids config csv file is stored.",
        required=True,
    )

    p.add_argument(
        "--out_dir",
        help="Path where the reports will be stored.",
        required=True,
    )

    p.add_argument(
        "--sr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the reports to be generated are for SR data.",
    )

    p.add_argument(
        "--add_js",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether some javascript should be added to the report for interaction with the index file.",
    )

    p.add_argument(
        "--block_if_exclude",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the ratings should be blocked after the first rating is set below the include threshold.",
    )

    p.add_argument(
        "--dataset",
        help="Name of the dataset to be used for the reports.",
        default=None,
    )

    p.add_argument(
        "--disable_bias_blur",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the bias field correction and blur artifact ratings should be disabled.",
    )
    args = p.parse_args()

    bids_list = csv_to_list(args.bids_csv)
    dataset = (
        args.bids_csv.split("/")[-1] if args.dataset is None else args.dataset
    )
    print_title("Generating reports")
    generate_report(
        bids_list,
        dataset=dataset,
        out_folder=args.out_dir,
        boundary=20,
        boundary_tp=20,
        ncols_ip=6,
        n_slices_tp=6,
        every_n_tp=4,
        annotate=False,
        cmap="Greys_r",
        do_index=args.add_js,
        is_sr=args.sr,
        block_if_exclude=args.block_if_exclude,
        disable_bias_blur=args.disable_bias_blur,
    )

    return 0


if __name__ == "__main__":
    main()
