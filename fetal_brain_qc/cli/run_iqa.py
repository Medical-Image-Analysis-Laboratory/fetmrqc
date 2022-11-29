""" Image quality assessment based on fetal-IQA. 

Based on the code from Junshen Xu at
https://github.com/daviddmc/fetal-IQA
"""

# Import libraries


def main():
    import os
    import torch
    import argparse
    import pandas as pd
    from pathlib import Path
    from fetal_brain_qc.utils import csv_to_list
    from fetal_brain_qc.fetal_IQA import eval_model, resnet34
    from fetal_brain_qc.definitions import FETAL_IQA_CKPT

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(
        description=(
            "Performs deep learning-based, slice-wise fetal brain"
            " image quality assessment."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--out-csv",
        help="Path where the IQA results will be stored.",
        required=True,
    )

    parser.add_argument(
        "--bids-csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )
    parser.add_argument(
        "--ckpt_path",
        help="Path to the checkpoint of the fetal IQA pytorch model.",
        default=FETAL_IQA_CKPT,
    )

    parser.add_argument(
        "--device",
        help="Device to be used for the deep learning model.",
        default="cuda:0",
    )

    args = parser.parse_args()
    bids_list = csv_to_list(args.bids_csv)

    print("Running slice-wise image quality assessment ... ")
    out_path = Path(args.out_csv)
    os.makedirs(out_path.parent, exist_ok=True)

    model = resnet34(pretrained=False, num_classes=3)
    model = torch.nn.DataParallel(model).to(args.device)
    checkpoint = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()

    pred_dict = {}
    for run in bids_list:
        im_path, mask_path = run["im"], run["mask"]
        name = Path(im_path).name
        pred_dict[name] = eval_model(im_path, mask_path, model, args.device)

    df = pd.DataFrame.from_dict(
        {
            (i, j): pred_dict[i][j]
            for i in pred_dict.keys()
            for j in pred_dict[i].keys()
        },
        orient="index",
    )

    df.to_csv(args.out_csv)
    return 0


if __name__ == "__main__":
    main()
