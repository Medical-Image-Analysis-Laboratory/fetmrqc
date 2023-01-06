from copy import deepcopy
from io import open
import json
import nibabel as ni
from .plotting import plot_mosaic, plot_mosaic_sr
from pathlib import Path
import matplotlib.pyplot as plt
from .data import IndividualTemplate
import random
import secrets
import string
import json
import shutil
import csv
import re
from pathlib import Path

import os
import time

REPORT_TITLES = [
    ("In-plane view", "ip1"),
    ("Through-plane view 1", "tp1"),
    ("Through-plane view 2", "tp2"),
]


def get_image_info(im_path):
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
        with open(im_json_path, "r") as f:
            config = json.load(f)
        im_info["field_strength"] = "oui"  # config["MagneticFieldStrength"]
    else:
        im_info["field_strength"] = "Unknown"

    return im_info


def read_report_snippet(in_file):
    """Add a snippet into the report. From MRIQC"""
    import os.path as op
    import re
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
):
    """From MRIQC"""

    import datetime
    from json import load

    # Now, the in_iqms file should be correctly named

    # Extract and prune metadata

    in_plots = [
        (REPORT_TITLES[i] + (read_report_snippet(v),))
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
    }

    tpl = IndividualTemplate()
    tpl.generate_conf(_config, out_path)


def generate_report(
    bids_list,
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
):
    tmp_report_dir = "tmp_report_plots"
    os.makedirs(out_folder, exist_ok=True)
    for run in bids_list:
        im_path = run["im"]
        mask_path = run["mask"]
        print(f"Processing {Path(im_path).name}")
        """Generate a report given an image path and mask"""
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
                boundary=boundary,
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
        im_info = get_image_info(im_path)
        out = individual_html(
            in_plots=out_plots,
            im_info=im_info,
            bids_name=run["name"],
            out_path=out_path,
            do_index=do_index,
        )
        plt.close()
    # Remove temporary directory for report generation
    shutil.rmtree(tmp_report_dir)
    return out
