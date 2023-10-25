import pandas as pd
import os
from fetal_brain_qc.qc_evaluation import METRICS, METRICS_SEG
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime
from fetal_brain_qc.definitions import FETMRQC20
import json


def load_dataset(dataset, first_iqm):
    df = pd.read_csv(dataset)
    xy_index = df.columns.tolist().index(first_iqm)

    train_x = df[df.columns[xy_index:]].copy()
    train_y = df[df.columns[:xy_index]].copy()
    return train_x, train_y


def get_rating(rating, is_class, class_threshold=1.0):
    """Format the rating: if it is a classification task,
    binarize the rating at the class_threshold
    """
    if is_class:
        if isinstance(rating, list):
            return [int(r > class_threshold) for r in rating]
        elif isinstance(rating, pd.DataFrame):
            return (rating > class_threshold).astype(int)
        else:
            return rating > class_threshold
    else:
        return rating


def model_name(is_class, iqms):
    type_ = "_class" if is_class else "_regr"
    if iqms == METRICS + METRICS_SEG:
        iqms = "_full"
    elif iqms == FETMRQC20:
        iqms = "_20"
    else:
        iqms = "_custom"
    return f"fetmrqc{iqms}{type_}.joblib"


DATASET = "/media/tsanchez/tsanchez_data/data/maddalena/iqms_full_no_test_no_avanto.csv"


def main():
    # Parser version of the code below
    import argparse

    parser = argparse.ArgumentParser("Train a FetMRQC model.")
    parser.add_argument(
        "--dataset",
        help="Path to the csv file dataset.",
        default=DATASET,
    )
    parser.add_argument(
        "--first_iqm",
        help="First IQM in the csv of the dataset.",
        default="centroid",
    )
    parser.add_argument(
        "--classification",
        help="Whether to perform classification or regression.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--regression",
        help="Whether to perform classification or regression.",
        dest="classification",
        action="store_false",
    )
    parser.add_argument(
        "--fetmrqc20",
        help="Whether to use FetMRQC20 IQMs.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--iqms_list",
        help="Custom list of IQMs to use. By default, all IQMs are used.",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--save_path", help="Where to save the model.", default="."
    )
    args = parser.parse_args()

    if args.iqms_list is not None:
        iqms = args.iqms_list
        print(f"Using custom IQMs: {iqms}")
    elif args.fetmrqc20:
        iqms = FETMRQC20
    else:
        iqms = METRICS + METRICS_SEG

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    out_path = os.path.join(
        args.save_path, model_name(args.classification, iqms)
    )
    train_x, train_y = load_dataset(args.dataset, args.first_iqm)

    model = (
        RandomForestClassifier()
        if args.classification
        else RandomForestRegressor()
    )
    rating = get_rating(train_y["rating"], args.classification)
    model.fit(train_x[iqms], rating)

    curr_time = datetime.now().strftime("%d%m%y_%H%M%S")
    dump(model, out_path)
    config = {
        "dataset": DATASET,
        "classification": args.classification,
        "timestamp": curr_time,
        "iqms": iqms,
    }

    with open(out_path.replace(".joblib", ".json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
