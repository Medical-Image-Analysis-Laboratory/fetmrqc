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
FetMRQC inference script
"""

# Import libraries


def main():
    import os
    import json
    import argparse
    import pandas as pd
    from pathlib import Path
    import joblib
    from fetal_brain_utils import print_title
    from fetal_brain_qc.definitions import (
        FETAL_FETMRQC_REG_CKPT,
        FETAL_FETMRQC_CLF_CKPT,
        FETAL_FETMRQC20_REG_CKPT,
        FETAL_FETMRQC20_CLF_CKPT,
        FETMRQC20,
    )
    from fetal_brain_qc.qc_evaluation import METRICS, METRICS_SEG

    parser = argparse.ArgumentParser(
        description=("Performs FetMRQC inference, given a pretrained model."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--iqms_csv",
        help="Path where the IQMs csv file is located.",
        required=True,
    )

    parser.add_argument(
        "--out_csv",
        help="CSV file where the predicted results will be stored.",
        required=True,
    )

    parser.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the fetal IQA pytorch model.",
        default=None,
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
        "--fetmrqc20",
        help="Whether to use FetMRQC20 IQMs.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--custom_model",
        help="Path to a custom model, trained using run_train_fetmrqc.py.",
        default=None,
        type=str,
    )

    args = parser.parse_args()
    bids_df = pd.read_csv(args.iqms_csv)

    if args.custom_model is not None:
        ckpt_path = args.custom_model
        json_path = args.custom_model.replace(".joblib", ".json")

        with open(json_path, "r") as f:
            json_dict = json.load(f)
        task = (
            "classification" if json_dict["classification"] else "regression"
        )
        print_title(f"Running a custom FetMRQC inference ({task}).")
        print(
            f"Dataset: {json_dict['dataset']}\nIQMs: {json_dict['iqms']}\n"
            f"Experiment done on: {json_dict['timestamp']}"
        )
        iqms = json_dict["iqms"]
    else:
        task = "classification" if args.classification else "regression"
        ckpt_path = args.ckpt_path

        if ckpt_path is None:
            if task == "classification":
                if not args.fetmrqc20:
                    ckpt_path = FETAL_FETMRQC_CLF_CKPT
                else:
                    ckpt_path = FETAL_FETMRQC20_CLF_CKPT
            elif task == "regression":
                if not args.fetmrqc20:
                    ckpt_path = FETAL_FETMRQC_REG_CKPT
                else:
                    ckpt_path = FETAL_FETMRQC20_REG_CKPT

        is20 = "20" if args.fetmrqc20 else ""
        print_title(f"Running FetMRQC{is20} inference ({task}).")
        if not args.fetmrqc20:
            iqms = METRICS + METRICS_SEG
        else:
            iqms = FETMRQC20

    model = joblib.load(ckpt_path)
    out_path = Path(args.out_csv)
    os.makedirs(out_path.parent, exist_ok=True)
    import pdb

    pdb.set_trace()
    test_x = bids_df[iqms]
    test_y = model.predict(test_x)

    if task == "classification":
        bids_df.insert(0, "fetmrqc", test_y.astype(int))
    else:
        bids_df.insert(0, "fetmrqc", test_y)

    bids_df.to_csv(out_path, index=False)
    print(f"Model predictions saved at {out_path.absolute() }.")
    return 0


if __name__ == "__main__":
    main()
