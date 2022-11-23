def main():
    import argparse
    from fetal_brain_qc.list_bids import list_bids
    from fetal_brain_qc.anon_bids import anonymize_bids_csv
    from fetal_brain_qc.definitions import MASK_PATTERN_LIST

    p = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory and tries to find corresponding masks given by "
            "`mask_patterns`. Then, saves all the found pairs of (LR series, masks) in "
            " a CSV file at `out_csv`"
        )
    )

    p.add_argument(
        "bids-dir",
        help="BIDS directory containing the LR series.",
    )

    p.add_argument(
        "--mask-patterns",
        help=(
            "List of patterns to find the LR masks corresponding to the LR series.\n "
            'Patterns will be of the form "sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}'
            '[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz", and the different fields will be '
            "substituted based on the structure of bids_dir. The code will attempt to find the mask corresponding "
            "to the first pattern, and if it fails, will attempt the next one, etc. If no masks are found, a warning "
            "message will be displayed for the given subject, session and run. "
        ),
        nargs="+",
        default=MASK_PATTERN_LIST,
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
    list_bids(
        args.bids_dir,
        args.mask_patterns,
        bids_csv=args.out_csv,
    )
    if args.anonymize_name:
        print(f"Anonymize name in {args.out_csv}.")
        anonymize_bids_csv(args.out_csv, out_bids_csv=args.out_csv)


if __name__ == "__main__":
    main()
