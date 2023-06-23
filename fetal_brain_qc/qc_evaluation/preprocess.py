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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    LabelBinarizer,
)
from sklearn.base import BaseEstimator, TransformerMixin, _OneToOneFeatureMixin
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
from abc import ABC, abstractmethod
from sklearn.preprocessing import QuantileTransformer

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
        X = X.drop(
            [
                field
                for field in self.drop
                if self.ignore is None or field not in self.ignore
            ],
            axis=1,
        )
        return X


class DropColumns(_FeatureSelection):
    """
    Wraps a data transformation to run only in specific
    columns [`source <https://stackoverflow.com/a/41461843/6820620>`_].

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


class GroupScaler(ABC, _OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Base class for scaling features by group.

    Parameters
    ----------
    groupby : str, optional
        Column name to group by. If None, it performs like the default scaler (not the group one).
    """

    def __init__(
        self,
        groupby=None,
        copy=True,
    ):
        self.groupby = groupby
        self.copy = copy

    @abstractmethod
    def _get_scaler(self):
        pass

    def fit(self, X, y=None):
        if self.groupby is None:
            self.scalers_ = self._get_scaler().fit(X)
        else:
            sites = X[[self.groupby]].values.squeeze()
            X_input = X.drop([self.groupby], axis=1)
            self.scalers_ = {}
            for group in set(sites):
                self.scalers_[group] = self._get_scaler().fit(
                    X_input[sites == group]
                )
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
                    self.scalers_[group] = self._get_scaler().fit(
                        X_input[sites == group]
                    )

                # Apply scaling
                X_input[sites == group] = self.scalers_[group].transform(
                    X_input[sites == group]
                )

            # Get sites back
            X_input[self.groupby] = sites
            return X_input


class GroupRobustScaler(GroupScaler):
    """Scale features using statistics that are robust to outliers in each group.

    Parameters
    ----------
    groupby : str, optional
        Column name to group by. If None, it performs like sklearn's RobustScaler.
    with_centering : bool, default=True
        If True, center the data before scaling.
    with_scaling : bool, default=True
        If True, scale the data to unit variance (or equivalently, unit standard deviation).
    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, default=(25.0, 75.0)
        Quantile range used to calculate scale_.
    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input is already a
        numpy array).
    unit_variance : bool, default=False
        If True, scale data so that normally distributed features have a variance of 1. This
        is achieved by dividing the data by the standard deviation of the features, after
        centering them. It does not shift/center the data, and thus does not destroy any
        sparsity.

    """

    def __init__(
        self,
        groupby=None,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance

        super().__init__(
            groupby=groupby,
        )

    def _get_scaler(self):
        return RobustScaler(
            with_centering=self.with_centering,
            with_scaling=self.with_scaling,
            quantile_range=self.quantile_range,
            copy=self.copy,
            unit_variance=self.unit_variance,
        )


class GroupStandardScaler(GroupScaler):
    """Standardize features by removing the mean and scaling to unit variance in each group.

    Parameters
    ----------
    groupby : str, optional
        Column name to group by.
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently, unit standard deviation).
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).
    """

    def __init__(
        self,
        groupby=None,
        *,
        with_mean=True,
        with_std=True,
        copy=True,
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

        super().__init__(
            groupby=groupby,
        )

    def _get_scaler(self):
        return StandardScaler(
            with_mean=self.with_mean,
            with_std=self.with_std,
            copy=self.copy,
        )


class GroupQuantileTransformer(GroupScaler):
    """Transform features using quantiles information in each group.

    Parameters
    ----------
    groupby : str, optional
        Column name to group by.
    n_quantiles : int, default=250
        Number of quantiles to be computed. It corresponds to the number of landmarks used to discretize the cumulative distribution function. If n_quantiles is larger than the number of samples, n_quantiles is set to the number of samples as a larger number of quantiles does not give a better approximation of the cumulative distribution function estimator.
    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are 'uniform' (default) or 'normal'.
    """

    def __init__(
        self,
        groupby=None,
        *,
        n_quantiles=250,
        output_distribution="uniform",
    ):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

        super().__init__(
            groupby=groupby,
        )

    def _get_scaler(self):
        return QuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
        )


class GroupRobustScaler_old(RobustScaler):
    def __init__(
        self,
        groupby=None,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
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


class GroupStandardScaler_old(StandardScaler):
    def __init__(
        self,
        groupby=None,
        *,
        copy=True,
        with_mean=True,
        with_std=True,
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
        self.group_scalers = (
            GroupRobustScaler,
            GroupStandardScaler,
            GroupQuantileTransformer,
        )

    def _check_scaler_group(self):
        """Checking whether the group in GroupScalerSelector and in self.scaler are consistent.
        This accepts:
            - GroupScalerSelector and self.scaler.groupby=None
            - GroupScalerSelecter and self.scaler with the same specified group
            - GroupScalerSelector with a group and self.scaler.groupby=None
        This raises an error:
            - GroupScalerSelector with a group and self.scaler with a different group
        """
        if isinstance(self.scaler, self.group_scalers):
            if self.scaler.groupby == self.group:
                pass
            elif (
                self.scaler.groupby != self.group
                and self.scaler.groupby is None
            ):
                self.scaler.groupby = self.group
            else:
                raise RuntimeError(
                    f"Inconsistent group in GroupScalerSelector (group={self.group}) and {self.scaler} (group={self.scaler.groupby})"
                )

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

        self._check_scaler_group()

        isobject = [
            o
            for o in X.columns[X.dtypes == object].tolist()
            if o != self.group
        ]
        if self.group is not None:
            assert (
                self.group in X.columns
            ), f"Group {self.group} not found in the dataframe."
            if not isinstance(self.scaler, self.group_scalers):
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
        dtypes = X.dtypes
        # GroupScalers need to have the dataframe passed, not just its content
        if isinstance(self.scaler, self.group_scalers):
            X = self.scaler.transform(X)
        else:
            X[X.columns] = self.scaler.transform(X[X.columns])
        # Restore dtypes. It's not clear to me why they were changed in the first place
        X = X.astype(dtypes)
        X = self.drop_group(X)

        return X


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

        self.drop = []
        self.dropped_cst = []
        self.dropped_corr = []

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

        if self.drop_cst:
            Xvar = X.var()
            self.dropped_cst = [col for col in X.columns[Xvar == 0.0]]
            self.drop += self.dropped_cst
            X = X[X.columns[Xvar > 0.0]]
        corr = X.corr(numeric_only=False)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # Find features with correlation greater than threshold
        self.dropped_corr = [
            column
            for column in upper.columns
            if any(abs(upper[column]) > self.threshold)
        ]
        self.drop += self.dropped_corr
        self.n_retained = nfeatures - len(self.drop)
        return self
