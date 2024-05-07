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


def main():
    import argparse
    from fetal_brain_qc.index import generate_index
    from fetal_brain_utils import print_title
    import numpy as np
    import random

    parser = argparse.ArgumentParser(
        description=(
            "Given a list of reports, generates an index.html file to navigate through them."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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

    parser.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the folders should be sorted before generating the index.",
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
        sort=args.sort,
    )

    return 0


if __name__ == "__main__":
    main()
