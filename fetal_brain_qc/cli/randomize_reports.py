def main():
    import argparse
    from fetal_brain_qc.randomize import randomize_reports

    p = argparse.ArgumentParser(
        description=(
            "Randomization of the reports located in `reports_path`. "
            "By default, the `n-reports` random reports will be sampled and "
            "`n-reports` different permutations of these reports will be saved as "
            "subfolders of `out-path` labelled as split_1 to split_<n-raters>. "
            "Each folder will contain an `ordering.csv` file with the randomized "
            "ordering to be used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "reports_path",
        help="Path where the reports are located",
    )

    p.add_argument(
        "out_path",
        help="Path where the randomized reports will be stored.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to control the randomization",
    )

    p.add_argument(
        "--n-reports",
        type=int,
        default=100,
        help="Number of reports that should be used in the randomized study.",
    )

    p.add_argument(
        "--n-raters",
        type=int,
        default=3,
        help="Number of raters (number of permutations of the data that must be computed).",
    )

    args = p.parse_args()
    randomize_reports(
        args.reports_path,
        args.out_path,
        args.n_reports,
        args.n_raters,
        args.seed,
    )

    return 0


if __name__ == "__main__":
    main()
