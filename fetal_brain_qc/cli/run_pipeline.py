def main():
    import argparse
    from fetal_brain_qc.definitions import (
        MASK_PATTERN,
        BRAIN_CKPT,
        FETAL_STACK_IQA_CKPT,
        FETAL_IQA_CKPT,
    )
    from fetal_brain_qc.metrics import DEFAULT_METRICS
    from fetal_brain_qc.utils import validate_inputs
    from pathlib import Path
    import os

    p = argparse.ArgumentParser(
        description=(
            "Given a `bids_dir`, lists the LR series in "
            " the directory and tries to find corresponding masks given by "
            "`mask_patterns`. Then, saves all the found pairs of (LR series, masks) in "
            " a CSV file at `out_path/bids.csv`"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        dest="bids_dir",
        help="BIDS directory containing the LR series.",
    )

    p.add_argument(
        dest="out_path",
        help="Path where the reports will be stored.",
    )

    p.add_argument(
        "--brain-extraction",
        help=(
            "Whether brain extraction should be run.\n"
            "If run, it will store masks at `out_path`/`mask_patterns[0]`."
        ),
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--run-qc",
        help=("Whether quality control should be run.\n"),
        action=argparse.BooleanOptionalAction,
        default=False,
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
        "--mask-patterns-base",
        help=(
            "Base folder(s) from which the LR masks must be listed.\n "
            "The method will look for masks at `mask-pattern-base`/`mask-patterns`. "
            "In this case, both `mask-patterns` and `mask-pattern-base` should be of the same length."
        ),
        nargs="+",
        default=None,
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

    p.add_argument(
        "--navigation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether the user should be able to freely navigate between reports. "
            "This is disabled for rating, to force user to process reports sequentially."
        ),
    )

    p.add_argument(
        "--metrics",
        help="Metrics to be evaluated.",
        default=DEFAULT_METRICS,
    )

    p.add_argument(
        "--device",
        help="Device to be used for the deep learning model.",
        default="cuda:0",
    )

    p.add_argument(
        "--ckpt_path_slice_iqa",
        help="Path to the checkpoint of the fetal IQA pytorch model (by Junshen Xu at MIT).",
        default=FETAL_IQA_CKPT,
    )

    p.add_argument(
        "--ckpt_path_stack_iqa",
        help="Path to the checkpoint of the fetal IQA tensorflow model (by Ivan Legorreta FNNDSC).",
        default=FETAL_STACK_IQA_CKPT,
    )

    p.add_argument(
        "--ckpt_path_brain_extraction",
        help="Path to the checkpoint of the MONAIfbs model.",
        default=BRAIN_CKPT,
    )

    args = p.parse_args()
    validate_inputs(args)

    # Creating various variables and paths
    os.makedirs(args.out_path, exist_ok=True)
    bids_csv = Path(args.out_path) / "bids.csv"

    # To be used only when args.brain_extraction is true
    masks_dir = Path(args.out_path) / "masks"

    mask_patterns_base = (
        [str(masks_dir)] if args.brain_extraction else args.mask_patterns_base
    )
    if mask_patterns_base:
        mask_patterns_base = " ".join(str(x) for x in mask_patterns_base)
    mask_patterns = " ".join(str(x) for x in args.mask_patterns)

    # BRAIN EXTRACTION
    if args.brain_extraction:
        print("Running Brain extraction")

        cmd = (
            f"qc_brain_extraction "
            f"{args.bids_dir} {masks_dir} "
            f"--ckpt_path {args.ckpt_path_brain_extraction} "
            f"--mask-pattern {mask_patterns}"
        )

        print(cmd)
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError("Brain extraction failed.")
    # BIDS FOLDER AND MASKS LIST
    print("Running list_bids.")

    cmd = (
        f"qc_list_bids_csv "
        f"{args.bids_dir} "
        f"--mask-patterns {mask_patterns} "
        f"--out-csv {bids_csv} "
    )
    cmd += (
        f"--mask-patterns-base {mask_patterns_base} "
        if mask_patterns_base
        else ""
    )
    cmd += "--anonymize-name" if args.anonymize_name else ""

    print(cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError("BIDS listing of data and masks failed.")

    # QC METRICS: PREPROCESSING AND EVALUATION
    if args.run_qc:
        metrics_csv = Path(args.out_path) / "metrics.csv"
        # Preprocessing skipped for now
        metrics = " ".join(str(x) for x in args.metrics)
        print(metrics_csv)
        cmd = (
            "qc_compute_metrics "
            f"--out-csv {metrics_csv} "
            f"--metrics {metrics} "
            f"--bids-csv {bids_csv} "
            f"--ckpt_path_slice_iqa {args.ckpt_path_slice_iqa} "
            f"--ckpt_path_stack_iqa {args.ckpt_path_stack_iqa} "
            f"--device {args.device}"
        )
    print(cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError("Quality metrics computation failed.")

    # GENERATING REPORTS
    cmd = f"qc_generate_reports {args.out_path} {bids_csv} --add-js"

    print(cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError("Report generation failed.")

    # RANDOMIZE REPORTS
    if args.randomize:
        import shutil

        print("Randomizing the reports.")
        raw_reports = Path(args.out_path) / "raw_reports/"
        os.makedirs(raw_reports)
        for f in os.listdir(args.out_path):
            f = Path(args.out_path) / f
            if os.path.isfile(f) and f.suffix == ".html":
                new_path = raw_reports / Path(f).name
                shutil.move(f, new_path)
        cmd = (
            "qc_randomize_reports "
            f"{raw_reports} {args.out_path} "
            f"--seed {args.seed} "
            f"--n-reports {args.n_reports} "
            f"--n-raters {args.n_raters}"
        )
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError("Report randomization failed.")

    print("GENERATING INDEX")
    cmd = f"qc_generate_index {args.out_path} --no-add-script-to-reports "
    cmd += ("--" if args.randomize else "--no-") + "use-ordering-file "
    cmd += ("--" if args.navigation else "--no-") + "navigation"

    print(cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError("Report indexing failed.")

    return 0


if __name__ == "__main__":
    main()
