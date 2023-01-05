import os
from pathlib import Path
from .utils import iter_bids
from bids import BIDSLayout
import csv


def list_bids(bids_dir, mask_pattern_list, bids_csv):
    """Given a bids directory `bids_dir` containing LR stacks of fetal brain,
    along with a list of patterns to the corresponding brain masks, create a
    csv file `bids_csv` listing the name, subject, session, run, LR_path
    and mask_path for all cases where a mask is found for a given LR_path.
    Ignores data without a mask available.

    More details about how mask_pattern_list is formatted is given in
    `run_list_and_anon_bids.py` and an example can be found in `definitions.py`.
    """
    bids_layout = BIDSLayout(bids_dir)

    file_list = list_masks(bids_layout, mask_pattern_list)

    with open(bids_csv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=file_list[0].keys())
        writer.writeheader()
        for data in file_list:
            writer.writerow(data)


def list_masks(bids_layout, mask_pattern_list):
    """Given a BIDSLayout and a list of mask_patterns,
    tries to find the masks that exist for each (subject, session, run)
    in the BIDS dataset, using the provided patterns.
    If no mask is found, the data isn't added to the returned file_list
    """
    from fetal_brain_qc.utils import fill_pattern

    file_list = []
    for sub, ses, run, out in iter_bids(bids_layout):

        paths = [
            fill_pattern(bids_layout, sub, ses, run, p)
            for p in mask_pattern_list
        ]
        for i, f in enumerate(paths):
            if os.path.exists(f):
                fname = Path(out).name.replace(".nii.gz", "")
                file_list.append(
                    {
                        "name": fname,
                        "sub": sub,
                        "ses": ses,
                        "run": run,
                        "im": out,
                        "mask": f,
                    }
                )
                break
            if i == len(paths) - 1:
                print(
                    f"WARNING: No mask found for sub-{sub}, ses-{ses}, run-{run}"
                )
    return file_list
