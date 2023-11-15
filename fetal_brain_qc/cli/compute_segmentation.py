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
""" Compute segmentation on low resolution clinical acquisitions.
By default the segmentation is computed using a nnUNet-v2 model pretrained on FeTA data,
but other methods can be used.
"""

# Import libraries

import pandas as pd
import os
from pathlib import Path
from fetal_brain_qc.preprocess import crop_input
from bids.layout.utils import parse_file_entities
from bids.layout.writing import build_path
from joblib import Parallel, delayed
import numpy as np
from fetal_brain_qc.definitions import NNUNET_CKPT

PATTERN = (
    "sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
    "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz"
)

PATTERN_BASE = (
    "sub-{subject}[/ses-{session}]/sub-{subject}"
    "[_ses-{session}][_acq-{acquisition}][_run-{run}]"
)
PATTERN_CROPPED = PATTERN_BASE + "_desc-cropped_T2w.nii.gz"
PATTERN_PROB_SEG = PATTERN_BASE + "_desc-proba_seg.npz"
PATTERN_SEG = PATTERN_BASE + "_seg.nii.gz"
PATTERN_PKL = PATTERN_BASE + "_T2w.pkl"
CONDA_PREFIX = os.environ["CONDA_PREFIX"]


def existing_seg(ents, out_path):
    """Check whether the segmentation has already been computed."""
    seg_path = out_path / build_path(ents, PATTERN_SEG)
    seg_proba_path = out_path / build_path(ents, PATTERN_PROB_SEG)
    seg_cropped_path = out_path / build_path(ents, PATTERN_CROPPED)
    seg_pkl_path = out_path / build_path(ents, PATTERN_PKL)
    if (
        os.path.exists(seg_path)
        and os.path.exists(seg_proba_path)
        and os.path.exists(seg_cropped_path)
        and os.path.exists(seg_pkl_path)
    ):
        return (
            ents.get("subject", None),
            ents.get("session", None),
            ents.get("run", None),
            seg_path,
            seg_proba_path,
        )
    else:
        return None


def compute_segmentations(
    bids_df, out_path, nnunet_res_path, nnunet_env_path, device
):
    """Compute segmentation on low resolution clinical acquisitions, using nnUNet
    The segmentation is crops the images, saves them to out_path
    and then runs the nnUNet inference and updates the DataFrame with the segmentation paths.

    Args:
        bids_df (pd.DataFrame): DataFrame containing the paths to the images and masks.
        out_path (str): Path to the output folder.
        nnunet_res_path (str): Path to the nnunet folder containing the model checkpoint.
        nnunet_env_path (str): Path to the environment in which nnunet was installed (from `conda env list`)
        device (str): Device on which to run the inference (e.g. "cuda:0")
    Returns:
        bids_df (pd.DataFrame): DataFrame containing the paths to the images, masks and segmentations.
    """
    # Check that model is nnUNet

    def mask_im(im, mask):
        """Crop the image and mask, save them to out_path and rename them for processing with nnUNet."""

        cropped = crop_input(
            im, mask, out_path, mask_image=True, save_mask=False
        )
        if cropped is not None:
            cropped = Path(cropped)
            # Rename the cropped images to the nnUNet format
            renamed = cropped.parent / (
                cropped.stem.split(".")[0] + "_0000.nii.gz"
            )
            os.rename(cropped, renamed)
        else:
            # Print a warning if the image could not be cropped
            print(f"WARNING: {im} could not be cropped.")

    print("Cropping images ...", end="")
    list_seg = []
    list_done = []
    for _, row in bids_df.iterrows():
        out = existing_seg(parse_file_entities(row["im"]), out_path)
        if out is None:
            list_seg.append((row["im"], row["mask"]))
        else:
            list_done.append(out)
    Parallel(n_jobs=4)(delayed(mask_im)(im, mask) for im, mask in list_seg)

    print(" done.")

    # Run nnUNet inference
    # /home/tsanchez/anaconda3/envs/nnunet/bin/

    if len(list_seg) > 0:
        cmd_path = os.path.join(nnunet_env_path, "bin", "nnUNetv2_predict")
        os.system(
            f"nnUNet_results={nnunet_res_path}  nnUNet_raw='' nnUNet_preprocessed='' {cmd_path} "
            f"-d 1 -i {out_path} -o {out_path} -c 2d -f 0 --save_probabilities "
            f"-device {device}"
        )
    else:
        print("All segmentations were already computed.")

    # Move the outputs to the corresponding BIDS folder which will contain
    # 1. The cropped images, 2. The segmentations 3. The probability maps 4. the pkl files.
    df = pd.DataFrame(columns=["sub", "ses", "run", "seg", "seg_proba"])
    for sub, ses, run, seg, seg_proba in list_done:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "sub": str(sub),
                        "ses": ses,
                        "run": run,
                        "seg": seg,
                        "seg_proba": seg_proba,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    for file in list(out_path.glob("*_T2w.nii.gz")):
        ents = parse_file_entities(file)
        sub = ents.get("subject", None)
        ses = ents.get("session", None)
        run = ents.get("run", None)
        sub = sub if isinstance(sub, str) else f"{sub:03d}"
        seg_path = out_path / build_path(ents, PATTERN)
        os.makedirs(seg_path.parent, exist_ok=True)
        file, seg_path = str(file), str(seg_path)
        seg_out = seg_path.replace("_T2w.nii.gz", "_seg.nii.gz")
        seg_proba = seg_path.replace("_T2w.nii.gz", "_desc-proba_seg.npz")
        os.rename(file, seg_out)
        os.rename(
            file.replace("_T2w.nii.gz", "_T2w.pkl"),
            seg_path.replace("_T2w.nii.gz", "_T2w.pkl"),
        )
        os.rename(file.replace("_T2w.nii.gz", "_T2w.npz"), seg_proba)
        os.rename(
            file.replace("_T2w.nii.gz", "_T2w_0000.nii.gz"),
            seg_path.replace("_T2w.nii.gz", "_desc-cropped_T2w.nii.gz"),
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "sub": str(sub),
                        "ses": ses,
                        "run": run,
                        "seg": seg_out,
                        "seg_proba": seg_proba,
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    # Merge the new dataframe with the bids_df and reorder the columns
    df[["ses", "run"]] = df[["ses", "run"]].applymap(
        lambda x: int(x) if x is not None else None
    )

    # Keep track of all the subjects for which the segmentation was not successful
    names = df[["sub", "ses", "run"]].apply(
        lambda x: "-".join([str(y) for y in x]), axis=1
    )
    not_included = bids_df[
        ~bids_df[["sub", "ses", "run"]]
        .apply(lambda x: "-".join([str(y) for y in x]), axis=1)
        .isin(names)
    ]["name"]
    if len(not_included) > 0:
        print(
            "Segmentation was not successful for the following images:"
            + str(" ".join(not_included))
        )

    bids_df = pd.merge(bids_df, df, on=["sub", "ses", "run"], how="left")
    new_columns = [
        col for col in bids_df.columns if col != "seg" and col != "seg_proba"
    ]
    new_columns.insert(bids_df.columns.get_loc("mask") + 1, "seg")
    new_columns.insert(bids_df.columns.get_loc("mask") + 2, "seg_proba")
    bids_df = bids_df.loc[:, new_columns]

    return bids_df


def load_and_run_segmentation(
    bids_csv, out_path, nnunet_res_path, nnunet_env_path, device
):
    """
    Loads the data from bids_csv, checks whether the segmentation has already been computed
    and if not, computes the segmentation, saves it to the <out_path> folder and updates the
    DataFrame with the segmentation paths.

    Args:
        bids_csv (str): Path to the bids_csv file.
        out_path (str): Path to the output folder.
        nnunet_res_path (str): Path to the nnunet folder containing the model checkpoint.
        nnunet_env_path (str): Path to the environment in which nnunet was installed (from `conda env list`)
    """
    if bids_csv.endswith(".csv"):
        df = pd.read_csv(bids_csv)
    elif bids_csv.endswith(".tsv"):
        df = pd.read_csv(bids_csv, sep="\t")
    else:
        raise ValueError("bids_csv must be either csv or tsv.")
    # This is needed b.c. in some cases we have subjects with integer only names and in some
    # other cases strings. So we're making everything strings and giving it a consistent naming.
    # This will fail when we have subjects where the subject is actually named sub-1 and not sub-001.
    df["sub"] = df["sub"].apply(
        lambda x: x
        if isinstance(x, str) and not x.isdigit()
        else f"{int(x):03d}"
    )
    df[["ses", "run"]] = df[["ses", "run"]].applymap(
        lambda x: int(x) if pd.notnull(x) else None
    )

    # Check if the dataframe has a seg column
    if "seg" in df.columns:
        # Iterate through the entries of the seg column and check that the path exit
        for _, row in df.iterrows():
            name, seg = row["name"], row["seg"]

            if not isinstance(seg, str) and np.isnan(seg):
                raise NotImplementedError(
                    f"Segmentation path for {name} is None. The segmentation should be rerun but this is currently not implemented."
                )
            if not os.path.exists(seg):
                raise ValueError(
                    f"'Seg' column found in {bids_csv}, but the file {seg} was not found."
                )
        if out_path is not None:
            print(
                "WARNING: out_path is specified but will be ignored as segmentation paths were provided in bids_csv."
            )
        print(
            "Segmentation already computed. All segmentations were found locally. Terminating."
        )
    else:
        df = compute_segmentations(
            df, out_path, nnunet_res_path, nnunet_env_path, device
        )

        # Save the dataframe to bids_csv or tsv depending on the extension of bids_csv
        if bids_csv.endswith(".csv"):
            df.to_csv(bids_csv, index=False)
        elif bids_csv.endswith(".tsv"):
            df.to_csv(bids_csv, index=False, sep="\t")

    return 0


def main():
    import os
    import argparse
    from pathlib import Path
    from fetal_brain_utils import print_title

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    p = argparse.ArgumentParser(
        description=(
            "Compute segmentation on low resolution clinical acquisitions, using a "
            "pretrained deep learning model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--bids_csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )

    p.add_argument(
        "--out_path",
        help="Path where the segmentations will be stored. (if not specified in bids_csv)",
    )

    p.add_argument(
        "--ckpt_path_model",
        help="Path to the checkpoint to be used.",
        default=None,
    )

    p.add_argument(
        "--nnunet_res_path",
        help="Path to the nnunet folder containing the checkpoint.",
        default=NNUNET_CKPT,
    )

    p.add_argument(
        "--nnunet_env_path",
        help="Path to the nnunet folder containing the checkpoint (from `conda env list`).",
        default=f"{CONDA_PREFIX}/envs/nnunet",
    )

    p.add_argument(
        "--device",
        help="Device to use for the nnUNet inference.",
        default="cuda",
        choices=["cuda", "cpu"],
    )

    args = p.parse_args()
    print_title("Running segmentation on the low-resolution images")
    out_path = Path(args.out_path).absolute()
    print(args.nnunet_res_path)
    load_and_run_segmentation(
        args.bids_csv,
        out_path,
        args.nnunet_res_path,
        args.nnunet_env_path,
        args.device,
    )
    return 0


if __name__ == "__main__":
    main()
