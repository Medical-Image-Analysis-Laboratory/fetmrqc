""" Bias correction using simpleITK. 

Based on the code from Michael Ebner (NiftyMIC) at
https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/utilities/n4_bias_field_correction.py
https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/application/correct_bias_field.py
"""

# Import libraries
import numpy as np
import os
import argparse
from fetal_brain_qc.preprocess import crop_input, correct_bias_field


def main():
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
        "--masked_path",
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
    from fetal_brain_qc.utils import csv_to_list

    bids_list = csv_to_list(args.bids_csv)

    print("Running N4ITK Bias Field Correction ... ")
    os.makedirs(args.out_path, exist_ok=True)
    # use_mask = True if args.filenames_masks is not None else False
    for run in bids_list:
        im_path = run["im"]
        print(f"\tProcessing {os.path.basename(im_path)}")
        mask_path = run["mask"]
        if args.cropping:
            im_path, mask_path = crop_input(
                im_path, mask_path, args.apply_mask, args.masked_path
            )

        correct_bias_field(
            im_path,
            mask_path,
            args.out_path,
            args.cropping,
            args.bias_field_fwhm,
            args.convergence_threshold,
            args.spline_order,
            args.wiener_filter_noise,
        )

    # for i, file_path in enumerate(args.filenames):

    #     if args.masked_path:
    #         if not use_mask:
    #             raise ValueError(
    #                 "dir-masked is set, but no masks were provided."
    #             )
    #         file_path_mask = args.filenames_masks[i] if use_mask else None

    print("Done!")

    return 0


if __name__ == "__main__":
    main()
