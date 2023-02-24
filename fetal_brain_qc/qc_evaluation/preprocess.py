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
"""Preprocessing transformers."""
import logging
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    LabelBinarizer,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from .qc_evaluation import REGRESSION_SCORING
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    GroupShuffleSplit,
)

LOG = logging.getLogger("mriqc_learn")
rng = np.random.default_rng()


class _FeatureSelection(BaseEstimator, TransformerMixin):
    """Base class to avoid reimplementations of transform."""

    def __init__(
        self,
        disable=False,
        ignore=None,
        drop=None,
    ):
        self.disable = disable
        self.ignore = ignore or tuple()
        self.drop = drop

    def transform(self, X, y=None):
        """Apply dimensionality reduction to X.
        X is masked.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        if self.disable or not self.drop:
            return X
        return X.drop(
            [field for field in self.drop if field not in self.ignore], axis=1
        )


class DropColumns(_FeatureSelection):
    """
    Wraps a data transformation to run only in specific
    columns [`source <https://stackoverflow.com/a/41461843/6820620>`_].

    Example
    -------

        >>> from mriqc_learn.models.preproces import DropColumns
        >>> tfm = DropColumns(columns=['duration', 'num_operations'])
        >>> # scaled = tfm.fit_transform(churn_d)

    """

    def __init__(self, drop):
        self.drop = drop
        self.disable = False
        self.ignore = tuple()

    def fit(self, X, y=None):
        return self


class PrintColumns(BaseEstimator, TransformerMixin):
    """Report to the log current columns."""

    def fit(self, X, y=None):
        cols = X.columns.tolist()
        LOG.warn(f"Features ({len(cols)}): {', '.join(cols)}.")
        return self

    def transform(self, X, y=None):
        return X


class GroupRobustScaler(RobustScaler):
    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
        groupby="site",
    ):
        self.groupby = groupby
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

    def fit(self, X, y=None):
        if self.groupby is None:
            self.scalers_ = RobustScaler(
                with_centering=self.with_centering,
                with_scaling=self.with_scaling,
                quantile_range=self.quantile_range,
                copy=self.copy,
                unit_variance=self.unit_variance,
            ).fit(X)
        else:
            sites = X[[self.groupby]].values.squeeze()
            X_input = X.drop([self.groupby], axis=1)
            self.scalers_ = {}
            for group in set(sites):
                self.scalers_[group] = RobustScaler(
                    with_centering=self.with_centering,
                    with_scaling=self.with_scaling,
                    quantile_range=self.quantile_range,
                    copy=self.copy,
                    unit_variance=self.unit_variance,
                ).fit(X_input[sites == group])
        return self

    def transform(self, X, y=None):
        if not self.scalers_:
            self.fit(X)

        if self.copy:
            X = X.copy()
        if self.groupby is None:
            X[X.columns] = self.scalers_.transform(X)
            return X
        else:
            sites = X[[self.groupby]].values.squeeze()
            X_input = X.drop([self.groupby], axis=1)

            for group in set(sites):
                if group not in self.scalers_:
                    # Yet unseen group
                    self.scalers_[group] = RobustScaler(
                        with_centering=self.with_centering,
                        with_scaling=self.with_scaling,
                        quantile_range=self.quantile_range,
                        copy=self.copy,
                        unit_variance=self.unit_variance,
                    ).fit(X_input[sites == group])

                # Apply scaling
                X_input[sites == group] = self.scalers_[group].transform(
                    X_input[sites == group]
                )

            # Get sites back
            X_input[self.groupby] = sites
            return X_input


class GroupStandardScaler(StandardScaler):
    def __init__(
        self,
        *,
        copy=True,
        with_mean=True,
        with_std=True,
        groupby="site",
    ):
        self.groupby = groupby
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X, y=None):
        if self.groupby is None:
            self.scalers_ = StandardScaler(
                with_mean=self.with_mean,
                with_std=self.with_std,
                copy=self.copy,
            ).fit(X)
        else:
            sites = X[[self.groupby]].values.squeeze()
            X_input = X.drop([self.groupby], axis=1)
            self.scalers_ = {}
            for group in set(sites):
                self.scalers_[group] = StandardScaler(
                    with_mean=self.with_mean,
                    with_std=self.with_std,
                    copy=self.copy,
                ).fit(X_input[sites == group])
        return self

    def transform(self, X, y=None):
        if not self.scalers_:
            self.fit(X)

        if self.copy:
            X = X.copy()
        if self.groupby is None:
            X[X.columns] = self.scalers_.transform(X)
            return X
        else:
            sites = X[[self.groupby]].values.squeeze()
            X_input = X.drop([self.groupby], axis=1)

            for group in set(sites):
                if group not in self.scalers_:
                    # Yet unseen group
                    self.scalers_[group] = StandardScaler(
                        with_mean=self.with_mean,
                        with_std=self.with_std,
                        copy=self.copy,
                    ).fit(X_input[sites == group])

                # Apply scaling
                X_input[sites == group] = self.scalers_[group].transform(
                    X_input[sites == group]
                )

            # Get sites back
            X_input[self.groupby] = sites
            return X_input


class NoiseWinnowFeatSelect(_FeatureSelection):
    """
    Remove features with less importance than a noise feature
    https://gist.github.com/satra/c6eb113055810f19709fa7c5ebd23de8

    """

    def __init__(
        self,
        n_winnow=10,
        use_classifier=False,
        n_estimators=1000,
        disable=False,
        k=1,
        ignore=("site",),
    ):
        self.ignore = ignore
        self.disable = disable
        self.n_winnow = n_winnow
        self.use_classifier = use_classifier
        self.n_estimators = n_estimators
        self.k = k

        self.importances_ = None
        self.importances_snr_ = None

    def fit(self, X, y=None, n_jobs=None):
        """Fit the model with X.
        This is the workhorse function.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        self.mask_ : array
            Logical array of features to keep
        """
        if self.disable:
            return self

        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        X_input = X.copy()
        columns_ = X_input.columns.tolist()

        self.drop = list(set(self.ignore).intersection(columns_))

        n_sample, n_feature = np.shape(X_input.drop(self.drop, axis=1))
        LOG.debug(f"Running Winnow selection with {n_feature} features.")

        if self.use_classifier:
            target_dim = np.squeeze(y).ndim
            multiclass = target_dim == 2 or np.unique(y).size > 2
            if multiclass and target_dim == 1:
                y = OrdinalEncoder().fit_transform(y)
            elif not multiclass:
                y = LabelBinarizer().fit_transform(y)

        if hasattr(y, "values"):
            y = y.values.squeeze()

        counter = 1
        noise_flag = True
        while noise_flag:
            # Drop masked features
            X = X_input.drop(self.drop, axis=1)
            # Add noise feature
            X["noise"] = _generate_noise(n_sample, y, self.use_classifier)

            # Initialize estimator
            clf = (
                ExtraTreesClassifier(
                    n_estimators=self.n_estimators,
                    criterion="gini",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_features="sqrt",
                    max_leaf_nodes=None,
                    min_impurity_decrease=1e-07,
                    bootstrap=True,
                    oob_score=False,
                    n_jobs=n_jobs,
                    random_state=None,
                    verbose=0,
                    warm_start=False,
                    class_weight="balanced",
                )
                if self.use_classifier
                else ExtraTreesRegressor(
                    n_estimators=self.n_estimators,
                    criterion="squared_error",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_leaf_nodes=None,
                    min_impurity_decrease=1e-07,
                    bootstrap=False,
                    oob_score=False,
                    n_jobs=n_jobs,
                    random_state=None,
                    verbose=0,
                    warm_start=False,
                )
            )
            clf.fit(X, y.reshape(-1))
            LOG.debug("done fitting once")
            importances = clf.feature_importances_
            drop_features = importances[:-1] <= (self.k * importances[-1])

            if not np.all(drop_features):
                self.drop += X.columns[:-1][drop_features].tolist()

            # fail safe
            if counter >= self.n_winnow:
                noise_flag = False

            counter += 1

        self.importances_ = importances[:-1]
        self.importances_snr_ = importances[:-1] / importances[-1]
        return self


def _generate_noise(n_sample, y, clf_flag=True):
    """
    Generates a random noise sample that is not correlated (<0.05)
    with the output y. Uses correlation if regression, and ROC AUC
    if classification
    """
    if clf_flag:
        return np.random.normal(loc=0, scale=1, size=(n_sample, 1))

    noise_corr = 1.0
    while noise_corr > 0.05:
        noise_feature = np.random.normal(loc=0, scale=10.0, size=(n_sample, 1))
        noise_corr = np.abs(
            np.corrcoef(noise_feature, y.reshape((n_sample, 1)), rowvar=0)[0][
                1
            ]
        )

    return noise_feature


class PassThroughScaler(BaseEstimator, TransformerMixin):
    """A scaler that does nothing, to be used
    as an option for GroupScalerSelector"""

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class GroupScalerSelector(BaseEstimator, TransformerMixin):
    """Applies a scaler in a pipeline. Flexibly handles scalers that require
    groups and basic sklearn scalers.

    If a group is passed, GroupScalerSelector drops it from the features X
    prior to fitting a sklearn scaler, and retains it for group scalers.
    After the computation of the transform, the group entry is dropped.

    Based on the code from
    https://stackoverflow.com/questions/63698484/how-to-test-preprocessing-combinations-in-nested-pipeline-using-gridsearchcv
    """

    def __init__(self, scaler=None, *, group=None):
        super().__init__()
        self.scaler = scaler if scaler is not None else PassThroughScaler()
        self.group = group
        if isinstance(self.scaler, (GroupRobustScaler, GroupStandardScaler)):
            assert (
                self.group == self.scaler.groupby
            ), f"ERROR: Grouping inconsistent between group={group} and scaler.groupby={self.scaler.groupby}"

    def drop_group(self, X):
        """If group is in the feature matrix X, drops it."""
        if self.group in X.columns:
            return X.drop(self.group, axis=1)
        return X

    def check_group(self, X):
        """Check whether the group should be dropped prior to fitting or transforming.
        Checks that the group is present in the feature dataframe X and that no entries
        in the dataframe have non number values, as this would make the fitting fail.
        """
        isobject = [
            o
            for o in X.columns[X.dtypes == object].tolist()
            if o != self.group
        ]

        if self.group is not None:
            assert (
                self.group in X.columns
            ), f"Group {self.group} not found in the dataframe."
            if not isinstance(
                self.scaler, (GroupRobustScaler, GroupStandardScaler)
            ):
                return self.drop_group(X)

        assert (
            len(isobject) == 0
        ), f"ERROR: Some non number values were found in column {isobject}"
        return X

    def fit(self, X, y=None):
        X = self.check_group(X)
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.check_group(X)
        scaled = self.scaler.transform(X)

        return self.drop_group(X)


class DropCorrelatedFeatures(_FeatureSelection):
    """ """

    def __init__(
        self,
        threshold=0.95,
        drop_cst=True,
        ignore="site",
    ):
        super().__init__()
        self.ignore = ignore
        self.drop_cst = drop_cst
        self.threshold = threshold

    def fit(self, X, y=None):
        """Fit the model with X.
        Parameters
        ----------
        X: pd.DataFrame (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        nfeatures = X.shape[1]
        if self.ignore is not None:
            X = X.drop(self.ignore, axis=1)
        Xvar = X.var()
        self.drop = [col for col in X.columns[Xvar == 0.0]]
        X = X[X.columns[Xvar > 0.0]]
        corr = X.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # Find features with correlation greater than threshold
        self.drop += [
            column
            for column in upper.columns
            if any(abs(upper[column]) > self.threshold)
        ]
        self.n_retained = nfeatures - len(self.drop)
        return self
