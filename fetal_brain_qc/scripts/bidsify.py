import json
from bids.layout.writing import build_path
from pathlib import Path
import re
import os
import glob
import gzip
import shutil
import csv
from fetal_brain_qc.utils import nested_defaultdict, iter_bids_dict


class BIDSdir:
    """Class to add files to a bids directory:
    Lists the current subjects and adds new ones after
    the last used subject id.
    """

    def __init__(self, bids_dir):
        self.bids_dir = bids_dir
        self._list_sub()
        if os.path.isdir(bids_dir):
            print(f"Found an existing directory at {bids_dir}.")
        else:
            print(f"Creating a new directory at {bids_dir}.")
            os.makedirs(bids_dir)

    def _list_sub(self):
        """List existing subjects, extracts the id of subjects
        defined as sub-<text><id>.
        """
        list_id = []
        # Extract the last digits in id
        # Match as little as possible with +?+
        p = re.compile(r".*?(\d+)")

        for f in os.listdir(self.bids_dir):
            if "sub-" in f:
                id_ = f[4:]
                id_ = int(p.findall(id_)[-1])
                list_id.append(id_)
        max_id = max(list_id)
        self._list_id = list_id
        self._sub_id = max_id + 1

    def _list_dir(self, base_dir, pattern):
        """List the directory to be copied into the bids_dir
        using a given regex pattern.
        Returns a nested dictionary with `sub`, `run`, `extension`
        as keys and the file path as the value.
        """
        # Recursively defined dictionary
        d = nested_defaultdict()
        p = re.compile(pattern)
        # List all files recursively
        for f in glob.glob(base_dir + "/**/*", recursive=True):
            # Fina a match for the pattern
            match = p.search(f)
            if match:
                regex_dict = match.groupdict()
                # Add file path to the dictionary in a nested fashion
                d.set(list(regex_dict.values()), f)
        ents = list(regex_dict.keys())
        # Return a true dictionary rather than a default dict
        return d.to_dict(), ents

    def get_file_path(self, subject, run, extension, acquisition):
        """Create the target file path from a given
        subject, run and extension.
        """
        if extension == "nii":
            extension = "nii.gz"
        ents = {
            "subject": subject,
            "session": None,
            "run": run,
            "datatype": "anat",
            "acquisition": acquisition,
            "suffix": "T2w",
            "extension": extension,
        }

        PATTERN = (
            self.bids_dir
            + "/sub-{subject}[/ses-{session}][/{datatype}]/sub-{subject}"
            "[_ses-{session}][_acq-{acquisition}][_run-{run}]_{suffix}.{extension}"
        )
        return build_path(ents, [PATTERN])

    def move_and_compress_nii(self, base_path, bids_path):
        """Copy a .nii file in base_path to a ".nii.gz" file
        in bids_path
        """
        with open(base_path, "rb") as f_in:
            with gzip.open(bids_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    def move_nii_gz(self, base_path, bids_path):
        """Copy a .nii.gz file in base_path to bids_path"""
        shutil.copyfile(base_path, bids_path)

    def move_and_edit_json(self, base_path, bids_path):
        " "
        with open(base_path, "r") as f_in:
            with open(bids_path, "w") as f_out:
                data = json.load(f_in)
                data["OriginalFilePath"] = str(base_path)
                json.dump(data, f_out)

    def copy2bids(self, base_path, bids_path):
        """Move a file to the bids directory:
        Creates the path if needed, moves it and if it as json,
        adds the original file_name as a field. If it is a
        .nii file, converts it to .nii.gz
        """
        base_path = Path(base_path)
        bids_path = Path(bids_path)
        if os.path.isfile(bids_path):
            raise RuntimeError(
                f"Target file {bids_path} already exists for {base_path}."
            )
        os.makedirs(bids_path.parent, exist_ok=True)

        suffix = "".join(base_path.suffixes)
        if suffix == ".nii":
            self.move_and_compress_nii(base_path, bids_path)
        elif suffix == ".nii.gz":
            self.move_nii_gz(base_path, bids_path)
        elif suffix == ".json":
            self.move_and_edit_json(base_path, bids_path)
        else:
            raise RuntimeError(f"Unknown extension {suffix} for {base_path}.")

    def dir2bids(self, base_dir, pattern, acquisition):
        """Copies and formats the files in base_dir into bids_dir.
        It uses a regex pattern that must list a subject, run and
        extension for each file that are to be mapped.

        Currently, this does not handle sessions.

        A typical example of pattern could be:
        '/(?P<sub>\w+)\/SERIES_(?P<run>\d+)\.(?P<ext>[\w\.]+)'
        ->It finds and copies files that satisfy a structure like
        "base_dir/{subject}/SERIES_{run}.{extension}".

        Currently, only ".nii", ".nii.gz" and ".json" are handled as
        extensions.
        """
        base_dict, ents = self._list_dir(base_dir, pattern)
        sub_dict = {}
        assert all([ent in ents for ent in ["sub", "run", "ext"]])
        print(f"Starting to move files at subject {self._sub_id}")
        for sub, run, ext, f in iter_bids_dict(base_dict, max_depth=3):
            # If subject not in dict: add it.
            if sub not in sub_dict.keys():
                sub_dict[sub] = f"{self._sub_id:03d}"
                self._sub_id += 1
            ents_curr = ents
            bids_path = self.get_file_path(
                sub_dict[sub], int(run), ext, acquisition
            )
            self.copy2bids(f, bids_path)
        # Write subject dictionary into participants.csv
        csv_path = self.bids_dir + "/participants.csv"
        if not os.path.isfile(csv_path):
            with open(csv_path, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["bids_id", "old_id"])
        with open(csv_path, "a") as csvfile:
            for k, v in sub_dict.items():
                writer = csv.writer(csvfile)
                writer.writerow([f"sub-{v}", k])


PATTERN_Z = "/(?P<sub>\w+)\/input_files\/\w*_(?P<run>\w+)\.(?P<ext>[\w\.]+)"
PATTERN_V = "/(?P<sub>\w+)\/SERIES_(?P<run>\d+)\.(?P<ext>[\w\.]+)"


def main():
    import argparse
    from fetal_brain_qc.list_bids import list_bids
    from fetal_brain_qc.anon_bids import anonymize_bids_csv
    from fetal_brain_qc.definitions import MASK_PATTERN, MANU_BASE, AUTO_BASE
    from fetal_brain_qc.utils import print_title
    import os

    p = argparse.ArgumentParser(
        description=(
            "Given a `base_dir` and a `bids_dir`, lists the LR series in "
            " `base_dir` and copies them as new subjects insides the `bids_dir`. "
            "It finds the files in `base_dir` according to a regex_pattern `pattern`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--base_dir",
        help="Directory containing the LR series to be BIDSified.",
        required=True,
        type=str,
    )

    p.add_argument(
        "--pattern",
        help=(
            "Pattern to be used to find the files in `base_dir`. "
            f"By default, you can specify pattern1 to use {PATTERN_Z} or pattern2 to use {PATTERN_V}. "
            "Other inputs will be interpreted as patterns to be used."
        ),
        required=True,
        type=str,
    )

    p.add_argument(
        "--bids_dir",
        help="BIDS directory containing the LR series.",
        required=True,
        type=str,
    )

    p.add_argument(
        "--acquisition",
        help="Name of the acquisition sequence used.",
        type=str,
    )

    args = p.parse_args()
    acq = None
    if args.pattern == "pattern_z":
        pattern = PATTERN_Z
        acq = "ssfse"
    elif args.pattern == "pattern_v":
        pattern = PATTERN_V
        acq = "usfe"
    else:
        pattern = args.pattern
    if acq:
        assert (
            args.acquisition is None
        ), f"args.acquisition must be None when using a prespecified pattern {args.pattern}"
    else:
        assert (
            args.acquisition is not None
        ), f"args.acquisition must be specified"
        acq = args.acquisition
    bids = BIDSdir(args.bids_dir)
    bids.dir2bids(args.base_dir, pattern, acq)


if __name__ == "__main__":
    main()
