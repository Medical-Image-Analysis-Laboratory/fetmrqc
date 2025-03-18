# FetMRQC: Quality control for fetal brain MRI
#
# Copyright 2023 Medical Image Analysis Laboratory (MIAL)
#
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
from copy import deepcopy
from io import open
import json
import nibabel as ni
from .plotting import plot_mosaic, plot_mosaic_sr
from pathlib import Path
import matplotlib.pyplot as plt
from .data import IndividualTemplate, IndividualSRTemplate
import random
import secrets
import string
import shutil
import csv
import re
import datetime
import os
import time


REPORT_TITLES = [
    ("In-plane view", "ip1"),
    ("Through-plane view 1", "tp1"),
    ("Through-plane view 2", "tp2"),
]

REPORT_TITLES_SR = [
    ("Quick view", "summary"),
    ("Axial view", "axial"),
    ("Sagittal view", "sagittal"),
    ("Coronal view", "coronal"),
]


def get_image_info(im_path, is_sr=False):
    """Extracting information for the report
    from the header of the nifti file as well
    as the json configuration file if it exists.
    """
    imh = ni.load(im_path).header
    im_json_path = im_path.replace("nii.gz", "json")

    im_info = dict(
        dim=imh["dim"][1:4],
        resolution=imh["pixdim"][1:4],
    )

    if os.path.isfile(im_json_path):
        try:
            config = json.load(open(im_json_path, "r"))
        except json.JSONDecodeError:
            print(f"Error reading {im_json_path}. Trying with pyjson5")
            try:
                import json5

                config = json5.load(open(im_json_path, "r"))
            except ImportError:
                print("json5 not installed. Skipping json file")
                config = {}
        im_info["field_strength"] = config.get(
            "MagneticFieldStrength", "Unknown"
        )
    else:
        im_info["field_strength"] = "Unknown"
    return im_info


def read_report_snippet(in_file):
    """Add a snippet into the report. From MRIQC"""
    import os.path as op
    from io import open  # pylint: disable=W0622

    is_svg = op.splitext(op.basename(in_file))[1] == ".svg"

    with open(in_file) as thisfile:
        if not is_svg:
            return thisfile.read()

        svg_tag_line = 0
        content = thisfile.read().split("\n")
        corrected = []
        for i, line in enumerate(content):
            if "<svg " in line:
                line = re.sub(' height="[0-9.]+[a-z]*"', "", line)
                line = re.sub(' width="[0-9.]+[a-z]*"', "", line)
                if svg_tag_line == 0:
                    svg_tag_line = i
            corrected.append(line)
        return "\n".join(corrected[svg_tag_line:])


def individual_html(
    in_plots,
    im_info,
    dataset="fetal_chuv",
    bids_name=None,
    out_path=None,
    do_index=False,
    sr=False,
    block_if_exclude=False,
    disable_bias_blur=False,
):
    """From MRIQC"""

    import datetime
    from json import load

    # Now, the in_iqms file should be correctly named

    # Extract and prune metadata

    if sr:
        titles = REPORT_TITLES_SR
    else:
        titles = REPORT_TITLES
    in_plots = [
        ((i,) + titles[i] + (read_report_snippet(v),))
        for i, v in enumerate(in_plots)
    ]
    date = datetime.datetime.now()
    timestamp = date.strftime("%Y-%m-%d, %H:%M")
    _config = {
        "dataset": dataset,
        "bids_name": bids_name,
        "timestamp": timestamp,
        "svg_files": in_plots,
        "im_info": im_info,
        "dim": im_info["dim"],
        "resolution": im_info["resolution"],
        "field_strength": im_info["field_strength"],
        "do_index": do_index,
        "block_if_exclude": block_if_exclude,
        "disable_bias_blur": disable_bias_blur,
    }

    tpl = IndividualTemplate() if not sr else IndividualSRTemplate()
    tpl.generate_conf(_config, out_path)


def generate_report(
    bids_list,
    dataset,
    out_folder=None,
    boundary=20,
    boundary_tp=20,
    ncols_ip=6,
    n_slices_tp=6,
    every_n_tp=4,
    annotate=False,
    cmap="Greys_r",
    do_index=False,
    is_sr=False,
    block_if_exclude=False,
    disable_bias_blur=False,
):

    tmp_report_dir = (
        f"tmp_report_plots_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    os.makedirs(out_folder, exist_ok=True)
    for i, run in enumerate(bids_list):
        im_path = run["im"]
        mask_path = run.get("mask", "")
        print(f"{i+1} - Processing {Path(im_path).name} as {run['name']}")
        """Generate a report given an image path and mask"""
        if True:
            mask_path = im_path.replace("T2w", "dseg")
        if mask_path == "":
            print(
                "WARNING: No mask was provided, using the binarization of the SR."
            )
        else:
            mask = ni.load(mask_path).get_fdata()
            if mask.sum() == 0:
                print(
                    f"\tWARNING: Empty mask {Path(mask_path).name}. Report generation skipped"
                )
                continue

        if is_sr:
            out_plots = plot_mosaic_sr(
                im_path,
                mask_path,
                boundary=10,
                ncols=ncols_ip,
                annotate=annotate,
                cmap=cmap,
                report_dir=tmp_report_dir,
            )
        else:
            out_plots = plot_mosaic(
                im_path,
                mask_path,
                boundary=boundary,
                boundary_tp=boundary_tp,
                ncols_ip=ncols_ip,
                n_slices_tp=n_slices_tp,
                every_n_tp=every_n_tp,
                annotate=annotate,
                cmap=cmap,
                report_dir=tmp_report_dir,
            )
        out_path = Path(out_folder) / (run["name"] + "_report.html")
        im_info = get_image_info(im_path, is_sr=is_sr)
        out = individual_html(
            in_plots=out_plots,
            im_info=im_info,
            dataset=dataset,
            bids_name=run["name"],
            out_path=out_path,
            do_index=do_index,
            sr=is_sr,
            block_if_exclude=block_if_exclude,
            disable_bias_blur=disable_bias_blur,
        )
        plt.close()
    # Remove temporary directory for report generation
    shutil.rmtree(tmp_report_dir)
    return out
