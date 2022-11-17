def main():
    import argparse
    from fetal_brain_qc.index import generate_index

    p = argparse.ArgumentParser()
    p.add_argument(
        "--report-path",
        help="Path where the reports are located",
    )
    p.add_argument(
        "--add-script-to-reports",
        type=bool,
        default=True,
        help="Whether some javascript should be added to the report for interaction with the index file.",
    )

    args = p.parse_args()
    generate_index(args.report_path)


if __name__ == "__main__":
    main()
