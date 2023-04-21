"""Test the classes GroupStandardScaler, GroupRobustScaler, PassThroughScaler and  GroupScalerSelector."""

import pytest
import numpy as np
import pandas as pd
from fetal_brain_qc.qc_evaluation.preprocess import (
    GroupStandardScaler,
    GroupRobustScaler,
    GroupScalerSelector,
    PassThroughScaler,
)


def get_df():
    """Create a small df as a running example."""
    df = pd.DataFrame(
        {
            "site": [
                "site1",
                "site1",
                "site1",
                "site2",
                "site2",
                "site2",
                "site3",
                "site3",
            ],
            "feature1": [1, 2, 3, 432, 4, 5, 6, -2],
            "feature2": [0.4, 0.3, -2, 0.3, 0.2, 0.1, 50, 100],
        }
    )
    return df


def test_passthrough():
    """Test the PassThroughScaler. Make sure that nothing is changed."""
    df = get_df()
    scaler = PassThroughScaler()
    df_scaled = scaler.fit_transform(df)
    assert df.equals(df_scaled)


def test_group_standard_scaler():
    """Test the GroupStandardScaler. Normalization should be (f - mean(f)) / std(f)"""
    df = get_df()
    scaler = GroupStandardScaler(groupby="site")
    df_scaled = scaler.fit_transform(df)

    for s in df_scaled.site:
        d = df_scaled[df_scaled["site"] == s].drop("site", axis=1)
        assert np.allclose(np.mean(d, axis=0), 0)
        assert np.allclose(np.var(d, axis=0), 1)


def test_group_robust_scaler():
    """Test the GroupStandardScaler. Normalization should be (f - median(f)) / IQR(f)"""
    df = get_df()
    scaler = GroupRobustScaler(groupby="site")
    df_scaled = scaler.fit_transform(df)

    for s in df_scaled.site:
        d = df_scaled[df_scaled["site"] == s].drop("site", axis=1)
        assert np.allclose(d.median(axis=0), 0)

        qlow = np.nanpercentile(d, 25, axis=0)
        qhigh = np.nanpercentile(d, 75, axis=0)
        assert np.allclose([u - l for u, l in zip(qhigh, qlow)], 1)


def test_group_scaler_selector_running():
    """Test the GroupScalerSelector. Make sure that the right scaler is used."""
    df = get_df()
    scaler = GroupScalerSelector(group="site", scaler=GroupStandardScaler())
    df_scaled = scaler.fit_transform(df.copy())
    df_scaled["site"] = df["site"]
    for s in df_scaled.site:
        d = df_scaled[df_scaled["site"] == s].drop("site", axis=1)
        assert np.allclose(np.mean(d, axis=0), 0)
        assert np.allclose(np.var(d, axis=0), 1)

    scaler = GroupScalerSelector(group="site", scaler=GroupRobustScaler())
    df_scaled = scaler.fit_transform(get_df())
    df_scaled["site"] = df["site"]
    for s in df_scaled.site:
        d = df_scaled[df_scaled["site"] == s].drop("site", axis=1)
        assert np.allclose(np.median(d, axis=0), 0)

        qlow = np.nanpercentile(d, 25, axis=0)
        qhigh = np.nanpercentile(d, 75, axis=0)
        assert np.allclose([u - l for u, l in zip(qhigh, qlow)], 1)

    scaler = GroupScalerSelector(group="site")
    df_scaled = scaler.fit_transform(df.copy())
    df_scaled.insert(0, "site", df["site"])
    assert df.equals(df_scaled)


def test_group_scaler_group_not_specified():
    df = get_df()

    # Test when both groups are None
    GroupScalerSelector(scaler=GroupStandardScaler()).fit_transform(
        df.copy().drop("site", axis=1)
    )
    # Test when only the group of GroupScalerSelector is specified
    GroupScalerSelector(
        group="site", scaler=GroupStandardScaler()
    ).fit_transform(df.copy())
    # Test when both groups are specified
    GroupScalerSelector(
        group="site", scaler=GroupStandardScaler(groupby="site")
    ).fit_transform(df.copy())

    # Check that the instruction raises a RuntimeError
    with pytest.raises(RuntimeError):
        GroupScalerSelector(
            group="subject", scaler=GroupStandardScaler(groupby="site")
        ).fit_transform(df.copy())
        GroupScalerSelector(
            scaler=GroupStandardScaler(groupby="site")
        ).fit_transform(df.copy())


def test_group_scaler_without_required_group():
    from sklearn.preprocessing import StandardScaler

    df = get_df()
    processed = GroupScalerSelector(
        group="site", scaler=StandardScaler()
    ).fit_transform(df)
    print(processed)
    #     This accepts:
    #     - GroupScalerSelector and self.scaler.groupby=None
    #     - GroupScalerSelecter and self.scaler with the same specified group
    #     - GroupScalerSelector with a group and self.scaler.groupby=None
    # This raises an error:
    #     - GroupScalerSelector with a group and self.scaler with a different group
