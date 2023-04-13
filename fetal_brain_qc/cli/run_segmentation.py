""" Compute segmentation on low resolution clinical acquisitions.
"""

# Import libraries

import pandas as pd
import os
from pathlib import Path
from fetal_brain_qc.preprocess import crop_input
from bids.layout.utils import parse_file_entities
from bids.layout.writing import build_path
PATTERN = (
    "sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
    "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz")

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
            print("WARNING: ckpt_path is specified but will be ignored in nnUNet.")

        cropped_img, seg_list, seg_proba_list = [], [], []

        # Crop the images and masks
        for im, mask in zip(bids_df["im"], bids_df["mask"]):
            print(f"Processing {im} {mask}")
            cropped = Path(crop_input(im, mask, out_path, mask_image=True, save_mask=False))
            # Rename the cropped images to the nnUNet format
            renamed = cropped.parent / (cropped.stem.split(".")[0] + "_0000.nii.gz")
            os.rename(cropped, renamed)
            cropped_img.append(renamed)
        
        # Run nnUNet inference
        os.system(f"nnUNetv2_predict -d 4 -i {out_path} -o {out_path} -c 2d -f 0 --save_probabilities")

        # Move the outputs to the corresponding BIDS folder which will contain
        # 1. The cropped images, 2. The segmentations 3. The probability maps 4. the pkl files. 
        for file in list(out_path.glob("*_T2w.nii.gz")):
            seg_path = out_path / build_path(parse_file_entities(file), PATTERN)
            os.makedirs(seg_path.parent, exist_ok=True)
            file, seg_path = str(file), str(seg_path)
            seg_out = seg_path.replace("_T2w.nii.gz","_seg.nii.gz")
            seg_proba =seg_path.replace("_T2w.nii.gz","desc-proba_seg.npz")
            os.rename(file, seg_out)
            os.rename(file.replace("_T2w.nii.gz","_T2w.pkl"), seg_path.replace("_T2w.nii.gz","_T2w.pkl"))
            os.rename(file.replace("_T2w.nii.gz","_T2w.npz"), seg_proba)
            os.rename(file.replace("_T2w.nii.gz","_T2w_0000.nii.gz"), seg_path.replace("_T2w.nii.gz","_desc-cropped_T2w.nii.gz"))     
            seg_list.append(seg_out)
            seg_proba_list.append(seg_proba)
    else:
        raise ValueError(f"Model {model} not supported. Please choose among {MODELS}")
    # add a new column with cropped_img to bids_df after the column named mask, with title seg
    bids_df.insert(bids_df.columns.get_loc("mask") + 1, "seg", seg_list)
    bids_df.insert(bids_df.columns.get_loc("seg") + 1, "seg_proba", seg_proba_list)
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
    # Check if the dataframe has a seg column
    if "seg" in df.columns:
        #Iterate through the entries of the seg column and check that the path exit
        for seg in df["seg"]:
            if not os.path.exists(seg):
                raise ValueError(f"'Seg' column found in {bids_csv}, but the file {seg} was not found.")
        if out_path is not None:
            print("WARNING: out_path is specified but will be ignored as segmentation paths were provided in bids_csv.")
        print("Segmentation already computed. All segmentations were found locally. Terminating.")
    else:
        df = compute_segmentations(df, out_path, model, ckpt_path)
        #Save the dataframe to bids_csv or tsv depending on the extension of bids_csv
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
        description=("Compute segmentation on low resolution clinical acquisitions."),
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
    load_and_run_segmentation(args.bids_csv, out_path, args.model, args.ckpt_path_model)
    return 0


if __name__ == "__main__":
    main()
