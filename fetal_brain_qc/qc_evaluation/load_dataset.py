# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Public API for MRIQC-learn datasets."""
from pathlib import Path
from pkg_resources import resource_filename as pkgrf

import numpy as np
from numpy.random import default_rng
import pandas as pd
import os
import json


def load_dataset(
    dataset="abide",
    split_strategy="random",
    test_split=0.2,
    seed=None,
    site=None,
    first_iqm="cjv",
    drop_nan=False,
):
    """Load default datasets."""
    if dataset not in ("abide", "ds030", "chuv100", "chuv_bcn"):
        raise ValueError(f"Unknown dataset <{dataset}>.")

    return load_data(
        path=Path(pkgrf("mriqc_learn.datasets", f"{dataset}.tsv")),
        split_strategy=split_strategy,
        test_split=test_split,
        seed=seed,
        site=site,
        first_iqm=first_iqm,
        drop_nan=drop_nan,
    )


def get_field_strength(im_str):
    """Take the metadata of the json file for a given image
    and extract the magnetic field strength, the model name,
    the TR and TE. and return it as a tuple of 4 values.
    """

    json_str = im_str.replace(".nii.gz", ".json")
    if os.path.isfile(json_str):
        with open(json_str, "r") as f:
            metadata = json.load(f)
        field = metadata["MagneticFieldStrength"]
        model = metadata["ManufacturersModelName"].replace("_", " ")
        TR = int(metadata["RepetitionTime"] * 1000)
        TE = int(metadata["EchoTime"] * 1000)
        return field, model, TR, TE
    else:
        print(im_str, "NO FILE FOUND")
        return None, None, None, None


def load_data(
    path=None,
    split_strategy="random",
    test_split=0.2,
    seed=None,
    site=None,
    first_iqm="cjv",
    drop_nan=False,
):
    """
    Load the ABIDE dataset.

    The loaded data are split into training and test datasets, and training
    and test are also divided in features and targets.

    Parameters
    ----------
    path : :obj:`os.pathlike`
        Whether to indicate a custom path were data are written in a TSV file.
    split_strategy : ``None`` or :obj:`str`
        How the data must be split into train and test subsets.
        Possible values are: ``"random"`` (default), ``"site"``, or ``None``/``"none"``.
    test_split : :obj:`float`
        Fraction of the dataset that will be split as test set when the
        split strategy is ``"random"``.
    seed : :obj:`int`
        A number to fix the seed of the random number generator
    site : :obj:`str`
        A site label indicating a particular site to be left out as test set.

    Returns
    -------
    (train_x, train_y), (test_x, test_y)
        The requested splits of the data

    """

    if site is not None:
        split_strategy = "site"

    if path is None:
        path = Path(pkgrf("mriqc_learn.datasets", "abide.tsv"))

    dataframe = pd.read_csv(path, index_col=None, delimiter=r"\s+")

    dataframe["field"] = dataframe[["im"]].applymap(get_field_strength)
    (
        dataframe["field"],
        dataframe["model"],
        dataframe["TR"],
        dataframe["TE"],
    ) = zip(*dataframe.field)
    # dataframe["im_size_x_n"], dataframe["im_size_z_n"]
    dataframe["site_field"] = dataframe.apply(
        lambda x: f"{x['site']} - {x['field']:.1f}", axis=1
    )
    dataframe["site_scanner"] = dataframe.apply(
        lambda x: f"{x['site']} - {x['model']}", axis=1
    )
    dataframe["vx_size"] = dataframe["im_size_vx_size"]
    # Return the position of the first IQM in the list
    cols = dataframe.columns.tolist()
    xy_index = cols.index(first_iqm)

    dataframe = dataframe[cols[:xy_index] + cols[-7:] + cols[xy_index:-7]]

    if drop_nan:
        dataframe = dataframe.dropna(axis=0)
        cols = dataframe.columns[xy_index:]
        types = {col: float if "nan" not in col else bool for col in cols}
        dataframe = dataframe.astype(types)
    if split_strategy is None or split_strategy.lower() == "none":
        return (
            dataframe[dataframe.columns[xy_index:]],
            dataframe[dataframe.columns[:xy_index]],
        ), (None, None)

    n = len(dataframe)
    rng = default_rng(seed)

    if split_strategy.lower() == "random":
        sample_idx = rng.integers(n, size=int(np.round(test_split * n)))
        test_df = dataframe.iloc[sample_idx]
        train_df = dataframe.drop(sample_idx)
    else:
        if site is None:
            sites = sorted(set(dataframe.site.unique()))
            site = sites[rng.integers(len(sites), size=1)[0]]

        sample = dataframe.site.str.contains(site)
        test_df = dataframe[sample]
        train_df = dataframe[~sample]

    return (
        train_df[dataframe.columns[xy_index:]],
        train_df[dataframe.columns[:xy_index]],
    ), (
        test_df[dataframe.columns[xy_index:]],
        test_df[dataframe.columns[:xy_index]],
    )
