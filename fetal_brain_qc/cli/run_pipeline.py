def main():
    import argparse
    from fetal_brain_qc.list_bids import list_bids
    from fetal_brain_qc.anon_bids import anonymize_bids_csv
    from fetal_brain_qc.definitions import MASK_PATTERN_LIST
    from fetal_brain_qc.report import generate_report
    from fetal_brain_qc.index import generate_index
    from fetal_brain_qc.randomize import randomize_reports
    import csv
    from pathlib import Path
    import os

    p = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory and tries to find corresponding masks given by "
            "`mask_patterns`. Then, saves all the found pairs of (LR series, masks) in "
            " a CSV file at `bids_csv`"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "bids-dir",
        help="BIDS directory containing the LR series.",
    )

    p.add_argument(
        "out-path",
        help="Path where the reports will be stored.",
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
        "--randomize",
        help=(
            "Whether the order of the reports should be randomized. "
            "The number of splits will be given by `n-raters` and the "
            "number of reports in each split by `n-reports`. The results will be stored "
            "in different subfolders of `out-path`"
        ),
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to control the randomization (to be used with randomize=True).",
    )

    p.add_argument(
        "--n-reports",
        type=int,
        default=100,
        help=(
            "Number of reports that should be used in the randomized study "
            "(to be used with randomize=True)."
        ),
    )

    p.add_argument(
        "--n-raters",
        type=int,
        default=3,
        help=(
            "Number of permutations of the data that must be computed "
            "(to be used with randomize=True)."
        ),
    )

    args = p.parse_args()

    print("Running list_bids.")
    bids_csv = Path(args.out_path) / "bids_csv.csv"
    os.makedirs(args.out_path, exist_ok=True)
    list_bids(
        args.bids_dir,
        args.mask_patterns,
        bids_csv=bids_csv,
    )
    if args.anonymize_name:
        print(f"Anonymize name in {bids_csv}.")
        anonymize_bids_csv(bids_csv, out_bids_csv=bids_csv)

    bids_list = []
    reader = csv.DictReader(open(bids_csv))
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

    if args.randomize:
        import shutil

        print("Randomizing the reports.")

        raw_reports = Path(args.out_path) / "raw_reports/"
        os.makedirs(raw_reports)
        for f in os.listdir(args.out_path):
            f = Path(args.out_path) / f
            if os.path.isfile(f):
                new_path = raw_reports / Path(f).name
                shutil.move(f, new_path)
        randomize_reports(
            raw_reports,
            args.out_path,
            args.n_reports,
            args.n_raters,
            args.seed,
        )
    print("Generating index.")
    generate_index(
        args.out_path,
        add_script_to_reports=False,
        use_ordering_file=args.randomize,
    )


if __name__ == "__main__":
    main()
