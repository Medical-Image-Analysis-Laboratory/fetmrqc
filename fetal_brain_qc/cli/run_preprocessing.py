""" Bias correction using simpleITK. 

Based on the code from Michael Ebner (NiftyMIC) at
https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/utilities/n4_bias_field_correction.py
https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/application/correct_bias_field.py
"""


def main():
    # Import libraries
    import numpy as np
    import os
    import argparse
    from fetal_brain_qc.preprocess import crop_input, correct_bias_field
    from fetal_brain_utils import csv_to_list, print_title
    import nibabel as ni
    from pathlib import Path
    from bids import BIDSLayout
    from fetal_brain_qc.utils import fill_pattern

    np.set_printoptions(precision=3)

    parser = argparse.ArgumentParser(
        description=(
            "Performs preprocessing: cropping of field of view"
            " followed by bias field correction (using N4ITK)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "out_path",
        help="Path where the bias corrected images will be stored.",
    )

    parser.add_argument(
        "bids_csv",
        help="Path where the bids config csv file is located.",
    )

    parser.add_argument(
        "--cropping",
        action=argparse.BooleanOptionalAction,
        help="Whether cropping should be performed.",
        default=True,
    )

    parser.add_argument(
        "--masked_cropped_path",
        type=str,
        help="Cropped/masked output directory.",
        default=None,
    )

    parser.add_argument(
        "--apply-mask",
        action=argparse.BooleanOptionalAction,
        help="Whether the image should be masked rather than only cropped.",
        default=False,
    )

    parser.add_argument(
        "--convergence-threshold",
        type=float,
        help="Specify the convergence threshold.",
        default=1e-6,
    )

    parser.add_argument(
        "--spline-order",
        type=int,
        help="Specify the spline order defining the bias field estimate.",
        default=3,
    )

    parser.add_argument(
        "--wiener-filter-noise",
        type=float,
        help="Specify the noise estimate defining the Wiener filter.",
        default=0.11,
    )

    parser.add_argument(
        "--bias-field-fwhm",
        type=float,
        help="Specify the full width at half maximum parameter characterizing "
        "the width of the Gaussian deconvolution.",
        default=0.15,
    )

    args = parser.parse_args()

    bids_list = csv_to_list(args.bids_csv)
    # Extract bids directory, build a layout and use it
    # to define where the output will be stored.
    bids_dir = bids_list[0]["im"].split("sub-")[0]
    layout = BIDSLayout(bids_dir)
    pattern = os.path.join(
        os.path.abspath(args.out_path),
        "sub-{subject}[/ses-{session}][/{datatype}]/",
    )

    print_title("Running N4ITK Bias Field Correction")

    for run in bids_list:
        im_path = run["im"]
        sub, ses, r = run["sub"], run["ses"], run["run"]
        out_path = fill_pattern(
            layout,
            sub,
            ses,
            r,
            pattern,
        )
        os.makedirs(out_path, exist_ok=True)

        print(f"\tProcessing {os.path.basename(im_path)}")
        mask_path = run["mask"]

        mask = ni.load(mask_path).get_fdata()
        if mask.sum() == 0:
            print(
                f"\tWARNING: Empty mask {Path(mask_path).name}. Report generation skipped"
            )
            continue
        if args.cropping:
            pattern_cropped = os.path.join(
                os.path.abspath(args.masked_cropped_path),
                "sub-{subject}[/ses-{session}][/{datatype}]/",
            )
            masked_cropped_path = fill_pattern(
                layout,
                sub,
                ses,
                r,
                pattern_cropped,
            )
            im_path, mask_path = crop_input(
                im_path, mask_path, args.apply_mask, masked_cropped_path
            )

        correct_bias_field(
            im_path,
            mask_path,
            out_path,
            args.apply_mask,
            args.bias_field_fwhm,
            args.convergence_threshold,
            args.spline_order,
            args.wiener_filter_noise,
        )

    print("Done!")

    return 0


if __name__ == "__main__":
    main()
