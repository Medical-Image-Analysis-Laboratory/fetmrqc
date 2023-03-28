from bs4 import BeautifulSoup as bs
from pathlib import Path
import numpy as np
import nibabel as ni
import copy
from pathlib import Path
import csv
import os
import operator
from collections import defaultdict
from functools import reduce
import json
import random


def squeeze_dim(arr, dim):
    if arr.shape[dim] == 1 and len(arr.shape) > 3:
        return np.squeeze(arr, axis=dim)
    return arr


def fill_pattern(bids_layout, sub, ses, run, pattern, suffix="T2w_mask"):

    query = bids_layout.get(subject=sub, session=ses, run=run)[0]
    acquisition = query.entities["acquisition"] if "acquisition" in query.entities else None
    ents = {
        "subject": sub,
        "session": ses,
        "run": run,
        "datatype": "anat",
        "acquisition": acquisition,
        "suffix": suffix,
    }
    return bids_layout.build_path(ents, pattern, validate=False)


def get_html_index(folder, use_ordering_file=False):
    """List all html files in the input `folder` or,
    if `use_ordering_file=True`, loads the ordering from
    `folder`/ordering.csv
    """
    folder = Path(folder)
    index_list = sorted(
        [
            f
            for f in Path(folder).iterdir()
            if f.is_file() and f.suffix == ".html" and "index" not in f.name
        ]
    )
    # raw_reports will not be ordered
    if "raw_reports" not in folder.name and (
        use_ordering_file and len(index_list) > 0
    ):
        ordering_file = Path(folder) / "ordering.csv"
        if not os.path.isfile(ordering_file):
            raise Exception(
                f"File ordering.csv not found at {ordering_file}. "
                "Did you mean to run with `--no-use-ordering-file`?"
            )

        reader = csv.DictReader(open(ordering_file))
        index_list = [Path(folder) / f["name"] for f in reader]
    elif len(index_list) > 0:
        if os.path.isfile(Path(folder) / "ordering.csv"):
            print(
                f"\tWARNING: ordering.csv was found but not used in {folder}.\n"
                f"\tDid you mean to run with --use-ordering-file?"
            )
    random.shuffle(index_list)
    return index_list


def add_message_to_reports(index_list):
    """Given a folder (`out_folder`) and a list of files in it (`index_list`),
    injects a javascript function into the html file to make it able to interact
    with the index.html file.
    """
    for file in index_list:
        # Parse HTML file in Beautiful Soup
        soup = bs(open(file), "html.parser")
        out = soup.find("script", type="text/javascript")
        in_str = out.string[1:]
        nspaces = len(in_str) - len(in_str.lstrip())
        newline = "\n" + " " * nspaces
        script_func = (
            f"{newline}$('#btn-download').click(function () {{{newline}"
            f"    window.parent.postMessage({{'message': 'rating done'}}, '*');"
            f"{newline}}});{newline}"
        )
        out.string = script_func + out.string
        with open(file, "w", encoding="utf-8") as f_output:
            f_output.write(str(soup))


def validate_inputs(args):
    """Check the validity of the arguments given
    in run_pipeline.py
    """
    if args.brain_extraction:
        assert (
            len(args.mask_patterns) == 1
        ), "A single mask_pattern must be provided when brain_extraction is enabled"
        assert args.mask_patterns_base is None, (
            "`mask_patterns_base` must be None when brain_extraction is enabled, "
            "as `out_path`/masks is used as the folder to store the outputs."
        )
    if args.randomize:
        raw_reports = Path(args.out_path) / "raw_reports/"
        assert not os.path.exists(raw_reports), (
            f"{args.out_path}/raw_reports path exists. Please define a different "
            "`out_path` for the experiment."
        )
        for i in range(args.n_raters):
            split = Path(args.out_path) / f"split_{i+1}/"
            assert not os.path.exists(split), (
                f"{split} path exists. Please define a different "
                "`out_path` for the experiment."
            )
