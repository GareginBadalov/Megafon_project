from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def data_transform(X):
    X['buy_time'] = pd.to_datetime(X['buy_time'], unit='s')
    X['Month'] = pd.DatetimeIndex(X['buy_time']).month
    return X


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)
