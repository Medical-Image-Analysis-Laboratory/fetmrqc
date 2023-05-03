import os
import argparse
import pandas as pd
from pathlib import Path
import nibabel as ni
from fetal_brain_utils import (
    csv_to_list,
    print_title,
    find_run_id,
    get_cropped_stack_based_on_mask,
)
from functools import partial
import re
import pdb


def stacks_selection(prepro_dir, img_list, mask_list):
    """Run the SVRTK-based function stacks-selection
    (requires a source installation of SVRTK)"""
    nstacks = len(img_list)
    # im_str = " ".join(img_list)
    # folder = Path(img_list[0]).parent
    # mask_str = " ".join(mask_list)
    # run_im = Path("/home/data") / im_file
    # run_m = Path("/home/data") / m_file
    data = " ".join(
        [str(Path("/home/data") / Path(im).name) for im in img_list]
    )
    masks = " ".join(
        [str(Path("/home/data") / Path(m).name) for m in mask_list]
    )
    cmd = (
        "docker run --rm "
        f"-v {prepro_dir}:/home/data "
        f"-v {prepro_dir}:/home/out/ "
        "svrtk_custom:auto-2.10 "
        f"/home/MIRTK/build/lib/tools/stacks-and-masks-selection {nstacks} "
        f"{data} {masks} "
    )
    print(cmd)
    os.system(cmd)


def crop_input(sub_ses_output, img_list, mask_list):
    boundary_mm = 15
    crop_path = partial(
        get_cropped_stack_based_on_mask,
        boundary_i=boundary_mm,
        boundary_j=boundary_mm,
        boundary_k=0,
    )
    im_list_c, mask_list_c, excluded = [], [], []

    for image, mask in zip(img_list, mask_list):
        im_file, mask_file = Path(image).name, Path(mask).name
        cropped_im_path = sub_ses_output / im_file
        cropped_mask_path = sub_ses_output / mask_file
        im, m = ni.load(image), ni.load(mask)

        imc = crop_path(im, m)
        maskc = crop_path(m, m)
        # Masking
        if imc is None:
            excluded.append(im_file)
        else:
            imc = ni.Nifti1Image(
                imc.get_fdata(), imc.affine, header=imc.header
            )
            maskc = ni.Nifti1Image(
                maskc.get_fdata(), imc.affine, header=imc.header
            )

            ni.save(imc, cropped_im_path)
            ni.save(maskc, cropped_mask_path)
            im_list_c.append(str(cropped_im_path))
            mask_list_c.append(str(cropped_mask_path))
    excluded = list(find_run_id(excluded).keys())
    if len(excluded) > 0:
        print(f"Excluded runs from image cropping: {excluded}")
    return im_list_c, mask_list_c, excluded


def is_run_included(run_curr, path, excluded):
    if os.path.isfile(path):
        pdb.set_trace()
        with open(path) as f:
            lines = f.readlines()
        out = lines[0].split(" ")
        n_reject = int(out[1]) + len(excluded)
        if n_reject > 0:
            return int(
                run_curr not in list(find_run_id(out[2:]).keys()) + excluded
            )
        else:
            return 1
    else:
        return "N/A"


def main():
    p = argparse.ArgumentParser(
        description=(
            "Exclude outlying stacks for each subject. Based on the code from NiftyMIC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--out_path",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    p.add_argument(
        "--bids_csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )
    p.add_argument(
        "--continue_run",
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

    metrics_dict = {}
    df_base = pd.DataFrame.from_dict(bids_list)
    df_base = df_base.set_index("name")
    os.makedirs(args.out_path, exist_ok=True)
    out_csv = Path(args.out_path) / "metrics.csv"
    # If a file is found, continue.
    if os.path.isfile(out_csv) and args.continue_run:
        print("\tCONTINUING FROM A PREVIOUSLY FOUND RUN.")
        df = pd.read_csv(out_csv).set_index("name")
        metrics_dict = df.to_dict(orient="index")
        metrics_dict = {
            k: {k2: v2 for k2, v2 in v.items() if k2 not in df_base.columns}
            for k, v in metrics_dict.items()
        }
    processed_sub_ses = []
    for run in bids_list:
        # Loading data
        name, sub, ses, run_ = (
            run["name"],
            run["sub"],
            run["ses"],
            int(run["run"]),
        )
        ses_str = f"{int(ses):02d}"
        sub_ses_output = (
            Path(args.out_path).resolve()
            / "cropped_input"
            / f"sub-{sub}/ses-{ses_str}/anat"
        )
        os.makedirs(sub_ses_output, exist_ok=True)

        if run["name"] in metrics_dict.keys():
            print(f"Subject {name} found in metrics.csv.")
            continue
        print(f"Processing subject and session {sub} {ses}")
        sel_sub = (df_base["ses"] == ses) & (df_base["sub"] == sub)
        sub_ses = f"sub-{sub}_ses-{ses}"
        path = sub_ses_output / "stats_summary.txt"

        if sub_ses not in processed_sub_ses:
            processed_sub_ses.append(sub_ses)
            img_list = list(df_base[sel_sub]["im"])
            mask_list = list(df_base[sel_sub]["mask"])
            img_list, mask_list, excluded = crop_input(
                sub_ses_output, img_list, mask_list
            )
            stacks_selection(sub_ses_output, img_list, mask_list)
            # stacks_and_masks_selection(img_list, mask_list)
        metrics_dict[run["name"]] = {
            "svrtk_qc": is_run_included(run_, path, excluded),
        }
        # Save the output throughout the training.
        df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        df = pd.concat([df_base, df], axis=1, join="inner")
        df.index.name = "name"
        df.to_csv(out_csv)

    return 0


if __name__ == "__main__":
    main()
