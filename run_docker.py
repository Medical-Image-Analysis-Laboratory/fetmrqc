# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This code is a wrapper to run the docker container of FetMRQC.
It is used to run the main calls to the pipeline, namely:
1. Reports generation using qc_reports_pipeline
2. Inference using qc_inference_pipeline
"""
import argparse
import os
from fetal_brain_qc.cli.build_run_parsers import (
    build_reports_parser,
    build_inference_parser,
)
from fetal_brain_qc.definitions import BRAIN_CKPT

MASK_PATTERN = (
    "sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
    "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.nii.gz"
)


def check_fixed_args(fixed_args: dict) -> None:
    """Check that the fixed arguments equal to their defaults.
    This is because not all arguments can be easily changed from the command line without
    being mounted explicitly on the docker.
    """
    defaults_dict = {"ckpt_path": [None, BRAIN_CKPT], "custom_model": [None]}
    for k, v in fixed_args.items():
        if v not in defaults_dict[k]:
            raise Warning(
                f"Argument {k} cannot be changed yet using the run_docker script. \n"
                f"It is fixed to {defaults_dict[k]}, but was passed as {v}. It is ignored."
            )


def check_paths(out_dict: dict, out_dir) -> None:
    """Check that the paths that should be part of the output are given
    as relative paths. Otherwise, they cannot be mounted on the docker.
    """

    for k, path in out_dict.items():
        if os.path.isabs(path):
            raise ValueError(
                f"Output paths ({out_dict.keys()}) should be relative paths with respect to the out_dir ({out_dir}), but the argument {k} was set to {path}."
            )


def run_reports(args: argparse.Namespace) -> None:
    """
    Run the reports pipeline using the given arguments.
    """
    # Create output directory
    reports_dir = "reports" if args.reports_dir is None else args.reports_dir
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, reports_dir), exist_ok=True)
    os.makedirs(
        os.path.dirname(os.path.join(args.out_dir, args.bids_csv)),
        exist_ok=True,
    )
    # Set paths
    out_docker = "/data/out"
    reports_dir_docker = os.path.join(out_docker, reports_dir)

    bids_csv = os.path.join(out_docker, args.bids_csv)
    bids_dir = os.path.abspath(args.bids_dir)
    masks_dir = os.path.abspath(args.masks_dir)
    out_dir = os.path.abspath(args.out_dir)

    os.makedirs(masks_dir, exist_ok=True)

    cmd = (
        f"docker run --rm -it "
        "--gpus all --ipc=host "  # Recommended by NVIDIA
        "--ulimit memlock=-1 --ulimit stack=67108864 "  # Recommended by NVIDIA
        f"-v {bids_dir}:/data/data "
        f"-v {masks_dir}:/data/masks "
        f"-v {out_dir}:{out_docker} "
        f"{args.docker_path} qc_reports_pipeline "
        f"--bids_dir /data/data "
        f"--masks_dir /data/masks "
        f"--reports_dir {reports_dir_docker} "
        f"--mask_pattern {args.mask_pattern} "
        f"--bids_csv {bids_csv} "
        f"--seed {args.seed} "
    )
    # Run command
    print(f"Running command: {cmd}")
    os.system(cmd)


def run_inference(args: argparse.Namespace) -> None:
    """
    Run the inference pipeline using the given arguments.
    """
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(
        os.path.dirname(os.path.join(args.out_dir, args.bids_csv)),
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(os.path.join(args.out_dir, args.iqms_csv)),
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(os.path.join(args.out_dir, args.out_csv)),
        exist_ok=True,
    )

    # Set paths
    out_docker = "/data/out"
    bids_csv = os.path.join(out_docker, args.bids_csv)
    iqms_csv = os.path.join(out_docker, args.iqms_csv)
    out_csv = os.path.join(out_docker, args.out_csv)
    bids_dir = os.path.abspath(args.bids_dir)
    masks_dir = os.path.abspath(args.masks_dir)
    seg_dir = os.path.abspath(args.seg_dir)
    out_dir = os.path.abspath(args.out_dir)

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    fetmrqc20 = (
        "--fetmrqc20_iqms " if args.fetmrqc20_iqms else "--no-fetmrqc20_iqms "
    )
    cmd = (
        f"docker run --rm -it "
        "--gpus all --gpus all --ipc=host "  # Recommended by NVIDIA
        "--ulimit memlock=-1 --ulimit stack=67108864 "  # Recommended by NVIDIA
        f"-v {bids_dir}:/data/data "
        f"-v {masks_dir}:/data/masks "
        f"-v {seg_dir}:/data/seg "
        f"-v {out_dir}:{out_docker} "
        f"{args.docker_path} qc_inference_pipeline "
        f"--bids_dir /data/data "
        f"--masks_dir /data/masks "
        f"--seg_dir /data/seg "
        f"--bids_csv {bids_csv} "
        f"--iqms_csv {iqms_csv} "
        f"--out_csv {out_csv} "
        f"{fetmrqc20} "
        f"--mask_pattern {args.mask_pattern} "
        f"--seed {args.seed} "
        f"--device {args.device} "
    )
    # Run command
    print(f"Running command: {cmd}")
    os.system(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "FetMRQC is a quality control tool for fetal brain MRI. "
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        help="Pipelines options",
        dest="command",
    )

    reports_parser = subparsers.add_parser(
        "reports",
        help="Creates reports for manual quality rating, given a BIDS dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_reports_parser(reports_parser)

    inference_parser = subparsers.add_parser(
        "inference",
        help="Run FetMRQC inference pipeline on a BIDS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_inference_parser(inference_parser)

    for p in [reports_parser, inference_parser]:
        p.add_argument(
            "--out_dir",
            help=(
                "Directory where the results will be stored. This folder is mounted on the docker, "
                "and all other outputs will be stored relatively to this folder."
            ),
            required=True,
        )

        p.add_argument(
            "--docker_path",
            help=("FetMRQC docker image to be used."),
            type=str,
            default="thsanchez/fetmrqc:latest",
        )

    args = parser.parse_args()

    fixed_args = {"ckpt_path": args.ckpt_path}

    if args.command == "reports":
        outputs_check = {
            "reports_dir": args.reports_dir,
            "bids_csv": args.bids_csv,
        }
        if args.reports_dir is None:
            outputs_check.pop("reports_dir", None)

    elif args.command == "inference":
        outputs_check = {
            "bids_csv": args.bids_csv,
            "iqms_csv": args.iqms_csv,
            "out_csv": args.out_csv,
        }
    else:
        raise ValueError(f"Unknown command {args.command}")
    check_fixed_args(fixed_args)
    check_paths(outputs_check, args.out_dir)

    if args.command == "reports":
        run_reports(args)
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
