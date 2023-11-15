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
from fetal_brain_qc.version import __version__, __url__
from fetal_brain_qc.cli.run_reports_pipeline import (
    build_parser as build_reports_parser,
)
from fetal_brain_qc.cli.run_inference_pipeline import (
    build_parser as build_inference_parser,
)


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
    )
    build_reports_parser(reports_parser)

    inference_parser = subparsers.add_parser(
        "inference",
        help="Run FetMRQC inference pipeline on a BIDS dataset",
    )
    build_inference_parser(inference_parser)

    args = parser.parse_args()
    import pdb

    pdb.set_trace()
    print(args)


if __name__ == "__main__":
    main()
