def main():
    import argparse
    from fetal_brain_qc.list_bids import list_bids
    from fetal_brain_qc.anon_bids import anonymize_bids_csv
    from fetal_brain_qc.definitions import MASK_PATTERN, MANU_BASE, AUTO_BASE
    import os

    p = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory and tries to find corresponding masks given by "
            "`mask_patterns`. Then, saves all the found pairs of (LR series, masks) in "
            " a CSV file at `out_csv`"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "bids_dir",
        help="BIDS directory containing the LR series.",
    )

    p.add_argument(
        "--mask-patterns",
        help=(
            "Pattern(s) to find the LR masks corresponding to the LR series.\n "
            'Patterns will be of the form "sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}'
            '[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz", and the different fields will be '
            "substituted based on the structure of bids_dir. The base directory from which the search will be run "
            "can be changed with `--mask-pattern-base`."
        ),
        nargs="+",
        default=[MASK_PATTERN],
    )

    p.add_argument(
        "--mask-pattern-base",
        help=(
            "Base folder(s) from which the LR masks must be listed.\n "
            "The method will look for masks at `mask-pattern-base`/`mask-patterns`. "
            "In this case, both `mask-patterns` and `mask-pattern-base` should be of the same length."
        ),
        nargs="+",
        default=None,
    )

    p.add_argument(
        "--out-csv",
        help="CSV file where the list of available LR series and masks is stored.",
        default="bids_csv.csv",
    )

    p.add_argument(
        "--anonymize-name",
        help=(
            "Whether an anonymized name must be stored along the paths in `out-csv`. "
            "This will determine whether the reports will be anonymous in the end."
        ),
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = p.parse_args()
    print("Running list_bids.")
    # Constructing patterns.
    if args.mask_pattern_base:
        print(args.mask_pattern_base, args.mask_patterns)
        assert len(args.mask_pattern_base) == len(
            args.mask_patterns
        ), "mask_pattern_base and mask_patterns have different lengths."
        mask_patterns = [
            os.path.join(os.path.abspath(b), m)
            for b, m in zip(args.mask_pattern_base, args.mask_patterns)
        ]
    else:
        mask_pattern_base = [MANU_BASE, AUTO_BASE]
        mask_patterns = [
            base + args.mask_patterns[0] for base in mask_pattern_base
        ]
    print(mask_patterns)
    list_bids(
        args.bids_dir,
        mask_patterns,
        bids_csv=args.out_csv,
    )
    if args.anonymize_name:
        print(f"Anonymize name in {args.out_csv}.")
        anonymize_bids_csv(args.out_csv, out_bids_csv=args.out_csv)


if __name__ == "__main__":
    main()
