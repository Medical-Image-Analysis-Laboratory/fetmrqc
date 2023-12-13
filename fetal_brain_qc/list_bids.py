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
import os
from pathlib import Path
from fetal_brain_utils import iter_bids
from bids import BIDSLayout
from fetal_brain_qc.definitions import MASK_PATTERN
import csv
import nibabel as ni


def create_sr_masks(bids_dir, mask_folder):
    """In some cases, the reconstructed images do not
    always have an associated brain mask (e.g. NeSVoR). It is then
    created simply by thresholding the image.
    """
    from fetal_brain_qc.utils import fill_pattern

    bids_layout = BIDSLayout(bids_dir, validate=False)
    out_pattern = os.path.join(mask_folder, MASK_PATTERN)
    for sub, ses, run, out in iter_bids(bids_layout):
        if "slice" in out:
            # Ignore slice folder.
            print("Ignoring slice folder")
            continue
        mask_out = fill_pattern(bids_layout, sub, ses, run, out_pattern)
        if not os.path.exists(os.path.dirname(mask_out)):
            os.makedirs(os.path.dirname(mask_out))
        im = ni.load(out)
        mask = im.get_fdata() > 0.0
        ni.save(
            ni.Nifti1Image(mask, affine=im.affine, header=im.header), mask_out
        )


def list_bids(bids_dir, mask_pattern_list, bids_csv, suffix="T2w"):
    """Given a bids directory `bids_dir` containing LR stacks of fetal brain,
    along with a list of patterns to the corresponding brain masks, create a
    csv file `bids_csv` listing the name, subject, session, run, LR_path
    and mask_path for all cases where a mask is found for a given LR_path.
    Ignores data without a mask available.

    More details about how mask_pattern_list is formatted is given in
    `run_list_and_anon_bids.py` and an example can be found in `definitions.py`.
    """
    bids_layout = BIDSLayout(bids_dir, validate=False)

    file_list = list_masks(bids_layout, mask_pattern_list, suffix=suffix)

    if len(file_list) == 0:
        error_msg = "Failed to list any elements. "
        if len(bids_layout.get_subjects()) == 0:
            error_msg += "BIDS Layout did not return any subjects. "
        else:
            error_msg += "Check that the mask pattern used is correct. "
        raise RuntimeError(error_msg)
    with open(bids_csv, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=file_list[0].keys())
        writer.writeheader()
        for data in file_list:
            writer.writerow(data)


def list_masks(bids_layout, mask_pattern_list, suffix="T2w"):
    """Given a BIDSLayout and a list of mask_patterns,
    tries to find the masks that exist for each (subject, session, run)
    in the BIDS dataset, using the provided patterns.
    If no mask is found, the data isn't added to the returned file_list
    """
    from fetal_brain_qc.utils import fill_pattern

    file_list = []
    for sub, ses, run, out in iter_bids(bids_layout, suffix=suffix):
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
                    f"Failed to match the pattern for sub-{sub}, ses-{ses}, run-{run}: Trying one more time with the get function... ",
                    end="",
                )
                out_m = bids_layout.get(
                    subject=sub,
                    session=ses,
                    run=run,
                    suffix="mask",
                    return_type="filename",
                )

                out_m = out_m[0] if len(out_m) > 0 else []
                if len(out_m) > 0:
                    print("Success!")
                    fname = Path(out).name.replace(".nii.gz", "")
                    file_list.append(
                        {
                            "name": fname,
                            "sub": sub,
                            "ses": ses,
                            "run": run,
                            "im": out,
                            "mask": out_m,
                        }
                    )
                else:
                    print("FAILED!")
    return file_list
