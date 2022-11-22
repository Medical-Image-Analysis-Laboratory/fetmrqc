import os
import random
import csv
from pathlib import Path
from .utils import get_html_index
import shutil


def randomize_and_sample_list(index_list, nreports=None):
    """Given a list of indexed reports, shuffles it and returns nreports."""
    random.shuffle(index_list)
    return index_list[:nreports] if nreports else index_list


def copy_files(reports_list: str, target_folder: str):
    """Copy the file from the reports_list to the target_folder."""
    target_folder = Path(target_folder)
    for report in reports_list:
        target = target_folder / Path(report).name
        shutil.copy(report, target)


def randomize_reports(reports_path, out_path, n_reports, n_raters, seed):
    """Randomization of the reports located in `reports_path`.
    By default, the `n-reports` random reports will be sampled and `n-reports`
    different permutations of these reports will be saved as subfolders of
    `out-path` labelled as split_1 to split_<n-raters>
    """
    random.seed(seed)
    reports_list = get_html_index(reports_path)
    reports_list = randomize_and_sample_list(reports_list, nreports=n_reports)

    out_path = Path(out_path)
    os.makedirs(out_path)

    for i in range(n_raters):
        out_folder = out_path / f"split_{i+1}"

        os.makedirs(out_folder)
        random.shuffle(reports_list)
        copy_files(reports_list, out_folder)

        reports_out = [[str(f.name)] for f in reports_list]
        with open(str(out_folder / "ordering.csv"), "w") as f:
            write = csv.writer(f)
            write.writerow(["name"])
            write.writerows(reports_out)
