""" Image quality assessment based on FNNDSC's implementation. 

Based on the code from Ivan Legorreta at
https://github.com/FNNDSC/pl-fetal-brain-assessment/tree/main
https://github.com/ilegorreta/Automatic-Fetal-Brain-Quality-Assessment-Tool
 (c) 2021 Fetal-Neonatal Neuroimaging & Developmental Science Center
                       Boston Children's Hospital
    http://childrenshospital.org/FNNDSC/
                            dev@babyMRI.org
"""


def main():
    import os
    import logging
    import argparse
    import pandas as pd
    import nibabel as ni
    from pathlib import Path

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    from fetal_brain_utils import csv_to_list, print_title
    from fetal_brain_qc.definitions import FETAL_STACK_IQA_CKPT
    from fetal_brain_qc.fnndsc_IQA import Predictor, fnndsc_preprocess

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description=(
            "Performs deep learning-based, stack-wise fetal brain"
            " image quality assessment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--out-csv",
        help="Path where the IQA results will be stored.",
        default="iqa_global_csv",
    )

    parser.add_argument(
        "--bids-csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )
    parser.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the fetal IQA pytorch model.",
        default=FETAL_STACK_IQA_CKPT,
    )

    parser.add_argument(
        "--device",
        help="Device to be used for the deep learning model.",
        default="cuda:0",
    )

    args = parser.parse_args()
    bids_list = csv_to_list(args.bids_csv)
    print_title("Running stack-wise image quality assessment")
    im_list, im_path_list, unrated_list = [], [], []
    for run in bids_list:
        # Loading data
        img = ni.load(run["im"]).get_fdata()
        mask = ni.load(run["mask"]).get_fdata().squeeze(-1)
        name = Path(run["im"]).name

        print(f"Preprocessing {name}")

        if mask.sum() > 0:  # Mask needed for preprocessing
            im_list.append(fnndsc_preprocess(img, mask))
            im_path_list.append(name)
        else:
            print(f"\tWARNING: empty mask for {name}")
            unrated_list.append({"filename": name, "quality": None})

    # Prediction
    df = Predictor(args.ckpt_path).predict(im_list, im_path_list)
    # Format the data without mask
    df_unrated = pd.DataFrame(unrated_list, columns=["filename", "quality"])
    # Save results
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    pd.concat([df, df_unrated], ignore_index=True).to_csv(args.out_csv)
    print("Done!")

    return 0


if __name__ == "__main__":
    main()
