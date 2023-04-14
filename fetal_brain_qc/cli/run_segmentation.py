""" Compute segmentation on low resolution clinical acquisitions.
"""

# Import libraries

import pandas as pd
import os
from pathlib import Path
from fetal_brain_qc.preprocess import crop_input
from bids.layout.utils import parse_file_entities
from bids.layout.writing import build_path
from joblib import Parallel, delayed
import time
import SimpleITK as sitk

PATTERN = (
    "sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
    "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz"
)

MODELS = ["nnUNet"]


def compute_segmentations(bids_df, out_path, model, ckpt_path):
    """Compute segmentation on low resolution clinical acquisitions.
    For model="nnUNet", the segmentation is crops the images, saves them to out_path
    and then runs the nnUNet inference and updates the DataFrame with the segmentation paths.

    Args:
        bids_df (pd.DataFrame): DataFrame containing the paths to the images and masks.
        out_path (str): Path to the output folder.
        model (str): Model to use for the segmentation (Currently, only nnUNet works).
        ckpt_path (str): Path to the model checkpoint.
    Returns:
        bids_df (pd.DataFrame): DataFrame containing the paths to the images, masks and segmentations.
    """
    # Check that model is nnUNet
    if model == "nnUNet":
        if ckpt_path is not None:
            print(
                "WARNING: ckpt_path is specified but will be ignored in nnUNet."
            )

        def mask_im(im, mask):
            """Crop the image and mask, save them to out_path and rename them for processing with nnUNet."""
            cropped = Path(
                crop_input(
                    im, mask, out_path, mask_image=True, save_mask=False
                )
            )
            # Rename the cropped images to the nnUNet format
            renamed = cropped.parent / (
                cropped.stem.split(".")[0] + "_0000.nii.gz"
            )
            os.rename(cropped, renamed)

        print("Cropping images ...", end="")
        Parallel(n_jobs=4)(
            delayed(mask_im)(im, mask)
            for im, mask in zip(bids_df["im"], bids_df["mask"])
        )
        print(" done.")

        # Run nnUNet inference
        os.system(
            f"nnUNetv2_predict -d 4 -i {out_path} -o {out_path} -c 2d -f 0 --save_probabilities"
        )

        # Move the outputs to the corresponding BIDS folder which will contain
        # 1. The cropped images, 2. The segmentations 3. The probability maps 4. the pkl files.
        df = pd.DataFrame(columns=["sub", "ses", "run", "seg", "seg_proba"])

        for file in list(out_path.glob("*_T2w.nii.gz")):
            ents = parse_file_entities(file)
            sub, ses, run = ents["subject"], ents["session"], ents["run"]
            sub = sub if isinstance(sub, str) else f"{sub:03d}"
            seg_path = out_path / build_path(ents, PATTERN)
            os.makedirs(seg_path.parent, exist_ok=True)
            file, seg_path = str(file), str(seg_path)
            seg_out = seg_path.replace("_T2w.nii.gz", "_seg.nii.gz")
            seg_proba = seg_path.replace("_T2w.nii.gz", "desc-proba_seg.npz")
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

    else:
        raise ValueError(
            f"Model {model} not supported. Please choose among {MODELS}"
        )

    # Merge the new dataframe with the bids_df and reorder the columns
    df[["ses", "run"]] = df[["ses", "run"]].astype(int)
    bids_df = pd.merge(bids_df, df, on=["sub", "ses", "run"])
    new_columns = [
        col for col in bids_df.columns if col != "seg" and col != "seg_proba"
    ]
    new_columns.insert(bids_df.columns.get_loc("mask") + 1, "seg")
    new_columns.insert(bids_df.columns.get_loc("mask") + 2, "seg_proba")
    bids_df = bids_df.loc[:, new_columns]

    return bids_df


def load_and_run_segmentation(bids_csv, out_path, model, ckpt_path):
    """
    Loads the data from bids_csv, checks whether the segmentation has already been computed
    and if not, computes the segmentation, saves it to the <out_path> folder and updates the
    DataFrame with the segmentation paths.

    Args:
        bids_csv (str): Path to the bids_csv file.
        out_path (str): Path to the output folder.
        model (str): Model to use for the segmentation (Currently, only nnUNet works).
        ckpt_path (str): Path to the model checkpoint.
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
        lambda x: x if isinstance(x, str) else f"{x:03d}"
    )

    # Check if the dataframe has a seg column
    if "seg" in df.columns:
        # Iterate through the entries of the seg column and check that the path exit
        for seg in df["seg"]:
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
        df = compute_segmentations(df, out_path, model, ckpt_path)
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
            "Compute segmentation on low resolution clinical acquisitions."
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
        "--model",
        help="Model to be used for the evaluation.",
        default="nnUNet",
        choices=MODELS,
    )
    p.add_argument(
        "--ckpt_path_model",
        help="Path to the checkpoint to be used.",
        default=None,
    )

    args = p.parse_args()
    print_title("Running segmentation on the low-resolution images")
    out_path = Path(args.out_path).absolute()
    load_and_run_segmentation(
        args.bids_csv, out_path, args.model, args.ckpt_path_model
    )
    return 0


if __name__ == "__main__":
    main()
