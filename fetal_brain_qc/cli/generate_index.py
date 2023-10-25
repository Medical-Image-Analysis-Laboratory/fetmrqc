def main():
    import argparse
    from fetal_brain_qc.index import generate_index
    from fetal_brain_utils import print_title
    import numpy as np
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports_dirs",
        nargs="+",
        help="Paths where the reports are located",
    )
    parser.add_argument(
        "--add_script_to_reports",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether some javascript should be added to the report for interaction with the index file.",
    )
    parser.add_argument(
        "--use_ordering_file",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether ordering.csv should be used to construct the ordering of index.html. "
            "The file should be located in the report-path folder."
        ),
    )
    parser.add_argument(
        "--navigation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether the user should be able to freely navigate between reports. "
            "This is disabled for rating, to force user to process reports sequentially."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to control the randomization (to be used with randomize=True).",
    )

    args = parser.parse_args()
    print_title("Generating index")
    np.random.seed(args.seed)
    random.seed(args.seed)
    generate_index(
        args.reports_dirs,
        args.add_script_to_reports,
        args.use_ordering_file,
        args.navigation,
    )

    return 0


if __name__ == "__main__":
    main()
