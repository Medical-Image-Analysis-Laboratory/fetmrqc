def main():
    import argparse
    from fetal_brain_qc.list_bids import list_bids
    from fetal_brain_qc.anon_bids import anonymize_bids_csv
    from fetal_brain_qc.definitions import MASK_PATTERN_LIST
    from fetal_brain_qc.report import generate_report
    from fetal_brain_qc.index import generate_index
    import csv

    p = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory and tries to find corresponding masks given by "
            "`mask_patterns`. Then, saves all the found pairs of (LR series, masks) in "
            " a CSV file at `bids_csv`"
        )
    )

    p.add_argument(
        "--bids-dir",
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
        "--bids-csv",
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

    p.add_argument(
        "-o",
        "--out-path",
        help="Path where the reports will be stored.",
    )

    args = p.parse_args()

    print("Running list_bids.")
    list_bids(
        args.bids_dir,
        args.mask_patterns,
        bids_csv=args.bids_csv,
    )
    if args.anonymize_name:
        print(f"Anonymize name in {args.bids_csv}.")
        anonymize_bids_csv(args.bids_csv, out_bids_csv=args.bids_csv)

    bids_list = []
    reader = csv.DictReader(open(args.bids_csv))
    for i, line in enumerate(reader):
        bids_list.append(line)

    print("Generating reports.")
    generate_report(
        bids_list,
        out_folder=args.out_path,
        boundary=20,
        boundary_tp=20,
        ncols_ip=6,
        n_slices_tp=6,
        every_n_tp=4,
        annotate=False,
        cmap="Greys_r",
        do_index=True,
    )

    print("Generating index.")
    generate_index(
        args.out_path,
        False,
    )


if __name__ == "__main__":
    main()
