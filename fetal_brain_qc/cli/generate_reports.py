def main():
    import json
    import argparse
    from fetal_brain_qc.report import generate_report

    p = argparse.ArgumentParser()

    p.add_argument(
        "-o",
        "--out-path",
        help="Path where the reports will be stored.",
    )

    p.add_argument(
        "--bids-csv",
        help="Path where the bids config csv file is located.",
    )

    p.add_argument(
        "--add-js",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether some javascript should be added to the report for interaction with the index file.",
    )

    args = p.parse_args()

    import csv

    bids_list = []
    reader = csv.DictReader(open(args.bids_csv))
    for i, line in enumerate(reader):
        bids_list.append(line)
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
        do_index=args.add_js,
    )


if __name__ == "__main__":
    main()
