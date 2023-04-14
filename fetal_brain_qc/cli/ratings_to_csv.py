import os


def get_last_json(ratings_list):
    """Utility function to get the last created file out of a list of files.
    Used to get only the json of the last rating.
    """
    if len(ratings_list) > 1:
        modification_time = [os.stat(f).st_mtime for f in ratings_list]
        sorted_ratings = [
            x for _, x in sorted(zip(modification_time, ratings_list))
        ]
        return sorted_ratings[-1]
    elif len(ratings_list) == 1:
        return ratings_list[0]
    else:
        return []


def rating_text(rating):
    """Convert the quality rating to a human readable category."""
    rating = float(rating)
    if rating < 0:
        raise ValueError("Invalid value. Rating should be in [0,4].")
    elif rating < 1:
        return "exclude"
    elif rating < 2:
        return "poor"
    elif rating < 3:
        return "acceptable"
    elif rating <= 4:
        return "excellent"
    else:
        raise ValueError("Invalid value. Rating should be in [0,4].")


def artifact_text(rating):
    """Convert the artifact rating to a human readable category."""
    rating = float(rating)
    if rating < 0.0:
        raise ValueError("Invalid value. Artifact should be in [0,3].")
    elif rating < 1.0:
        return "low"
    elif rating < 2.0:
        return "moderate"
    elif rating <= 3.0:
        return "high"
    else:
        raise ValueError("Invalid value. Rating should be in [0,3].")


def main():
    import argparse
    from fetal_brain_utils import csv_to_list
    import pandas as pd
    import re
    import json

    p = argparse.ArgumentParser(
        description=(
            "Given a `ratings_dir`, and a `bids_csv`, formats the ratings into "
            " a single csv file containing all information. "
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        dest="ratings_dir",
        help="Directory containing the ratings.",
    )

    p.add_argument(
        dest="bids_csv",
        help="CSV file where the list of available LR series and masks is stored.",
    )

    p.add_argument(
        "--out_csv",
        help="CSV file where the ratings will be stored (default: `<ratings_dir>/ratings.csv`).",
    )

    args = p.parse_args()

    bids_list = csv_to_list(args.bids_csv)
    df_base = pd.DataFrame.from_dict(bids_list)
    df_base = df_base.set_index("name")
    files = [
        os.path.join(args.ratings_dir, f)
        for f in sorted(os.listdir(args.ratings_dir))
        if f.endswith(".json")
    ]
    # files_progress = [f for f in files if "progress" in f]
    files_ratings = [f for f in files if "progress" not in f]
    df_base = df_base.assign(
        ratings_json=df_base.apply(lambda row: [], axis=1)
    )

    for f in files_ratings:
        sub = re.findall(r"_(sub-\w+?)_", str(f))[0]
        df_base.at[sub, "ratings_json"].append(f)

    unrated = list(df_base[df_base["ratings_json"].apply(len) == 0].index)
    if len(unrated) > 0:
        unrated = ["'" + str(x) + "_report.html'" for x in unrated]
        unrated_str = ", ".join(unrated)
        error = f"{len(unrated)} subjects were not rated. Aborting. The unrated subjects are \n{unrated_str}"
        # raise RuntimeError(error)
        print(f"WARNING: {error}")

    df_base["ratings_json"] = df_base["ratings_json"].apply(get_last_json)
    df_base = df_base.assign(
        artifacts=df_base.apply(lambda row: [], axis=1),
        selected_slices=df_base.apply(lambda row: [], axis=1),
    )

    for (idx, row) in df_base.T.items():
        if row["ratings_json"] != []:
            with open(row["ratings_json"], "r") as f:
                data = json.load(f)

            for k, v in data.items():
                # Format the slices selected to a list
                if k == "selected_slices":
                    if v == "":
                        df_base.at[idx, k] = []
                    else:
                        df_base.at[idx, k] = [int(f) for f in v.split(",")]

                else:
                    df_base.at[idx, k] = v
    # Convert some entries to float
    df_base = df_base.dropna()
    df_base[["rating", "fetal_motion", "bias_field"]].astype(float)
    # Add human readable categories to the ratings.
    df_base["rating_text"] = df_base["rating"].apply(rating_text)
    df_base["fetal_motion_text"] = df_base["fetal_motion"].apply(artifact_text)
    df_base["bias_field_text"] = df_base["bias_field"].apply(artifact_text)
    # Count the number of slices selected with artifacts.
    df_base["nselected"] = df_base["selected_slices"].apply(len)

    # Shuffling of the columns
    df_base = df_base[
        [
            "sub",
            "ses",
            "run",
            "rating",
            "rating_text",
            "orientation",
            "fetal_motion",
            "fetal_motion_text",
            "bias_field",
            "bias_field_text",
            "artifacts",
            "nselected",
            "selected_slices",
            "comments",
            "time_sec",
            "timestamp",
            "dataset",
            "im",
            "mask",
            "ratings_json",
        ]
    ]

    if args.out_csv is None:
        df_base.to_csv(os.path.join(args.ratings_dir, "ratings.csv"))
    else:
        df_base.to_csv(args.out_csv)


if __name__ == "__main__":
    main()
