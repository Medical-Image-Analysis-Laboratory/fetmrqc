def main():
    import argparse
    from fetal_brain_qc.index import generate_index

    p = argparse.ArgumentParser()
    p.add_argument(
        "--report-path",
        nargs="+",
        help="Path where the reports are located",
    )
    p.add_argument(
        "--add-script-to-reports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether some javascript should be added to the report for interaction with the index file.",
    )
    p.add_argument(
        "--use-ordering-file",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether ordering.csv should be used to construct the ordering of index.html. "
            "The file should be located in the report-path folder."
        ),
    )

    args = p.parse_args()
    generate_index(
        args.report_path,
        args.add_script_to_reports,
        args.use_ordering_file,
    )


if __name__ == "__main__":
    main()
