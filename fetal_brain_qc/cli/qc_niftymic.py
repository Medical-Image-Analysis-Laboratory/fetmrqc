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
    import os
    import argparse
    import pandas as pd
    from pathlib import Path
    from fetal_brain_utils import csv_to_list, print_title
    from fetal_brain_qc.metrics import LRStackMetrics
    import pdb

    p = argparse.ArgumentParser(
        description=(
            "Exclude outlying stacks for each subject. Based on the code from NiftyMIC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--out-csv",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    p.add_argument(
        "--bids-csv",
        help="Path where the bids config csv file is located.",
        required=True,
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

    args = p.parse_args()
    bids_list = csv_to_list(args.bids_csv)
    print_title("Running QC evaluation")

    lr_metrics = LRStackMetrics(
        metrics=["mask_volume"],
    )

    metrics_dict = {}
    df_base = pd.DataFrame.from_dict(bids_list)
    df_base = df_base.set_index("name")
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    # If a file is found, continue.
    if os.path.isfile(args.out_csv) and args.continue_run:
        print("\tCONTINUING FROM A PREVIOUSLY FOUND RUN.")
        df = pd.read_csv(args.out_csv).set_index("name")
        metrics_dict = df.to_dict(orient="index")
        metrics_dict = {
            k: {k2: v2 for k2, v2 in v.items() if k2 not in df_base.columns}
            for k, v in metrics_dict.items()
        }
    for run in bids_list:
        # Loading data
        name = Path(run["im"]).name
        if run["name"] in metrics_dict.keys():
            print(f"Subject {name} found in metrics.csv.")
            continue
        print(f"Processing subject {name}")
        metrics_dict[run["name"]] = lr_metrics.evaluate_metrics(
            run["im"], run["mask"]
        )

        # Save the output throughout the training.
        df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        df = pd.concat([df_base, df], axis=1, join="inner")
        df.index.name = "name"
        df.to_csv(args.out_csv)

    def get_median(input):
        median_volume = input.median()
        return (input > 0.7 * median_volume).astype(int)

    df[["niftymic_qc"]] = (
        df[["sub", "ses", "mask_volume"]]
        .groupby(["sub", "ses"])
        .transform(get_median)
    )
    df.to_csv(args.out_csv)
    return 0


if __name__ == "__main__":
    main()
