import pytest
from fetal_brain_utils.utils import get_mask_path, iter_dir, iter_bids
from fetal_brain_qc.cli.run_pipeline import main as run_pipeline

from bids import BIDSLayout
from pathlib import Path
import json
import os
import shutil
FILE_DIR = Path(__file__).parent.resolve()
BIDS_DIR = FILE_DIR / "data"
MASKS_DIR = FILE_DIR / "data/derivatives/masks"


# Write a test for the main function of run_pipeline.py where you call the parser and check that everything is run as expected.
def test_run_pipeline():
    """Test that the main function runs without error."""
    # Note: Using simulated data, the results are not
    # meaningful, but the pipeline should run without error.
    
    out_dir = FILE_DIR / "test_run_pipeline"
    
    if out_dir.exists():
        shutil.rmtree(out_dir)
    run_pipeline(
        [
            "--bids_dir",
            str(BIDS_DIR),
            "--out_path",
            str(out_dir),
            "--brain_extraction",
            "--n_reports",
            "2",
            "--n_raters",
            "1",
            "--run_qc"
        ]
    )

    # Recursively list the files in out_dir relatively from out_dir
    # in a deterministic order
    list_files = []
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            list_files.append(os.path.join(root, file).replace(str(out_dir) + "/", ""))
    list_files.sort()

    # If out_dir exists, recursively delete the folder using shutil.rmtree
    if out_dir.exists():
        shutil.rmtree(out_dir)

    with open(FILE_DIR / "output/test_run_pipeline.txt", "r") as f:
        txt = f.readlines()

    assert list_files == [l.replace("\n", "") for l in txt]
