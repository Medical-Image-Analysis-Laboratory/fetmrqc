from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numpy as np


class CustomStratifiedGroupKFold(StratifiedGroupKFold):
    """Extend stratified group k fold to be able to define splits as
    a proportion of the data to be used for training.
    """

    def __init__(self, train_size=0.8, shuffle=False, random_state=None):
        assert train_size >= 0 and train_size <= 1.0
        if train_size >= 0.5:
            n_splits = int(1 / (1 - train_size))
            self.reverse = False
        else:
            n_splits = int(1 / train_size)
            self.reverse = True
        super().__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            if self.reverse:
                train_index = indices[test_index]
                test_index = indices[np.logical_not(test_index)]
            else:
                train_index = indices[np.logical_not(test_index)]
                test_index = indices[test_index]
            yield train_index, test_index
