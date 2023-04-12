import pytest
from fetal_brain_utils.utils import get_mask_path, iter_dir, iter_bids
from fetal_brain_qc.cli.run_pipeline import main as run_pipeline
from fetal_brain_qc.cli.run_pipeline import main as main_pipeline
from bids import BIDSLayout
from pathlib import Path
import json
import os
import shutil
import pandas as pd
import numpy as np
FILE_DIR = Path(__file__).parent.resolve()
BIDS_DIR = FILE_DIR / "data"
MASKS_DIR = FILE_DIR / "data/derivatives/masks"

def compare_dataframes(df1, df2, numeric_tolerance=1e-6):
    # compare the string columns
    string_cols = df1.select_dtypes(include=['object']).columns
    string_compare = all(df1[col].equals(df2[col]) for col in string_cols)

    # compare the numeric columns
    numeric_cols = df1.select_dtypes(include=['number']).columns
    num_compare = np.isclose(df1[numeric_cols], df2[numeric_cols], rtol=numeric_tolerance).all().all()
    print(string_compare, num_compare)
    # return the comparison results
    return string_compare and num_compare


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
    df_out = pd.read_csv(out_dir / "metrics.csv")
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
    df_ref = pd.read_csv(FILE_DIR / "output/metrics.csv")
    
    # Compare df_ref and df_out and make sure that strings match and numbers are close
    assert compare_dataframes(df_out, df_ref)

def mock_iqms(n_rows=50):
    # import LRStackMetrics and get all available metrics
    from fetal_brain_qc.metrics import LRStackMetrics
    cols = list(LRStackMetrics().get_all_metrics())
    cols += [iqm +"_nan" for iqm in cols]
    cols = ["name","sub_ses","run","rating"]+cols
    # create a dataframe with the columns
    df = pd.DataFrame(columns=cols)
    sub_ses = 1
    while df.shape[0] < n_rows:
        # get a random number between 3 and 8
        n_runs = np.random.randint(3,8)
        sub_ses_str = f"sub-{sub_ses:03d}_ses-01"
        for run in range(n_runs):
            # create a row with a random string for name, sub_ses_str, run+1, and a random rating between 1 and 4
            # and random positive values for the metrics
            row = [np.random.bytes(5).hex(),sub_ses_str,run+1,np.random.randint(1,5)]+np.random.rand(len(cols)-4).tolist()
            df.loc[df.shape[0]] = row
            if df.shape[0] >= n_rows:
                break
        sub_ses += 1
    return df


def test_run_iqms():
    n_entries = 200
    df = mock_iqms(n_entries)
    assert df.shape[0] == n_entries
    # save the dataframe to a output/test_run_iqms.csv file
    df.to_csv(FILE_DIR / "output/test_run_iqms.csv",index=False)
    # call the command line interface qc_evaluate_qc using the saved csv file and save the results to "out/out_file" (create the folder)
    from fetal_brain_qc.cli.run_eval_qc_sacred import run_experiment
    # delete the out_iqms folder if it exists
    if (FILE_DIR / "out_iqms").exists():
        shutil.rmtree(FILE_DIR / "out_iqms")
    dataset = {"dataset_path": str(FILE_DIR / "output/test_run_iqms.csv"), "first_iqm": "centroid"}
    experiment = {
        "type": "regression",
        "classification_threshold": None,
        "metrics": "base",
        "scoring": "neg_mae",
    }

    cv = {
        "outer_cv": {"cv": "CustomStratifiedGroupKFold", "group_by": "sub_ses", "train_size": 0.5, "binarize_threshold":1},
        "inner_cv": {"cv": "CustomStratifiedGroupKFold", "group_by": "sub_ses", "train_size": 0.5, "binarize_threshold":1},
    }

    parameters = {
        "drop_correlated__threshold": [0.95, 1.0],
        "pca": ["passthrough"], 
        "noise_feature": ["passthrough"],  
        "scaler__scaler": ["StandardScaler()", "RobustScaler()"],
        "model": ["LinearRegression()"],
    }

    nested_score, metrics_list = run_experiment(
        dataset,
        experiment,
        cv,
        parameters
    )
    assert all(["test_"+m in nested_score.keys() for m in ["r2","neg_mae","neg_median_ae","spearman","acc","f1","prec","rec","roc_auc"]])
    assert all([nested_score["test_"+m].shape==(2,) for m in ["r2","neg_mae","neg_median_ae","spearman","acc","f1","prec","rec","roc_auc"]])
    
    

    