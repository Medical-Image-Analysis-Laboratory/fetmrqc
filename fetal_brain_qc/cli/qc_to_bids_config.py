"""
FetMRQC inference script
"""

# Import libraries


def main():
    import os
    import json
    import argparse
    import pandas as pd
    from pathlib import Path
    from fetal_brain_utils import print_title

    parser = argparse.ArgumentParser(
        description=(
            "Given FetMRQC inference, creates a config file with accepted stacks to give as input to SR reconstruction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bids-csv",
        help="Path where the bids config csv file is located.",
        required=True,
    )

    parser.add_argument(
        "--out-config",
        help="Path where the config will be stored.",
        required=True,
    )

    args = parser.parse_args()
    bids_df = pd.read_csv(args.bids_csv)
    out_config = Path(args.out_config)

    print_title("Creation of the SR reconstruction config file.")

    config_tmp = {}
    # Iterate the dataframe and create the raw config file
    for _, row in bids_df.iterrows():
        if row["fetmrqc_qc_passed"]:
            sub = row["sub"]
            ses = f"{row['ses']:02d}" if row["ses"] is not None else None
            stack = row["run"]
            sub_ses = f"sub-{sub}_ses-{ses}"
            if sub_ses not in config_tmp:
                config_tmp[sub_ses] = {
                    "sr-id": 1,
                    "sub": sub,
                    "session": ses,
                    "stacks": [stack],
                }
            else:
                config_tmp[sub_ses]["stacks"].append(stack)

    # Format the final config file to have
    # sub as index, and a dictionary for each session
    config = {}
    for conf in config_tmp.values():
        if conf["sub"] not in config.keys():
            sub = conf.pop("sub")
            config[sub] = [conf]
        else:
            conf.pop("sub")
            config[sub].append(conf)

    with open(out_config, "w") as f:
        json.dump(config, f, indent=4)
    return 0


if __name__ == "__main__":
    main()
