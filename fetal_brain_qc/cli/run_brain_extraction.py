def main():
    import argparse
    import os
    from bids import BIDSLayout
    from fetal_brain_qc.definitions import MASK_PATTERN, BRAIN_CKPT
    from fetal_brain_qc.brain_extraction import (
        bidsify_monaifbs,
        run_brain_extraction,
    )
    from fetal_brain_utils import iter_bids, print_title

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
        "bids_dir",
        help="BIDS directory containing the LR series.",
    )

    p.add_argument(
        "masks_dir",
        help="Root of the BIDS directory where brain masks will be stored.",
    )

    p.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the MONAIfbs model.",
        default=BRAIN_CKPT,
    )

    p.add_argument(
        "--mask-pattern",
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
    # Listing all files
    files_paths = [out for _, _, _, out in iter_bids(bids_layout)]
    # Create a tmp directory for the output of monaifbs segmentation
    masks_tmp = os.path.join(args.masks_dir, "tmp")
    run_brain_extraction(files_paths, masks_tmp, brain_ckpt=args.ckpt_path)
    # Move files to their definitive location
    bidsify_monaifbs(bids_layout, mask_pattern, masks_tmp)

    return 0


if __name__ == "__main__":
    main()
