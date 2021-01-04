import numpy as np
import pandas as pd
from sklearn import metrics


def root_mean_squared_error(y_test, y_pred):
    """Evaluate forecasts"""
    if not isinstance(y_pred, pd.Series):
        raise NotImplementedError(
            "multi-step forecasting horizons with multiple cutoffs/windows "
            "are not supported yet")

    # select only test points for which we have made predictions
    y_pred = y_pred.dropna()
    if not np.all(np.isin(y_pred.index, y_test.index)):
        raise IndexError("Predicted time points are not in test set")
    y_test = y_test.loc[y_pred.index]

    return metrics.mean_squared_error(y_test, y_pred, squared=False)


def mean_absolute_error(y_test, y_pred):
    """Evaluate forecasts"""
    if not isinstance(y_pred, pd.Series):
        raise NotImplementedError(
            "multi-step forecasting horizons with multiple cutoffs/windows "
            "are not supported yet")

    # select only test points for which we have made predictions
    y_pred = y_pred.dropna()
    if not np.all(np.isin(y_pred.index, y_test.index)):
        raise IndexError("Predicted time points are not in test set")
    y_test = y_test.loc[y_pred.index]

    return metrics.mean_absolute_error(y_test, y_pred)


def mean_absolute_percentage_error(y_test, y_pred):
    """Evaluate forecasts"""
    if not isinstance(y_pred, pd.Series):
        raise NotImplementedError(
            "multi-step forecasting horizons with multiple cutoffs/windows "
            "are not supported yet")

    # select only test points for which we have made predictions
    y_pred = y_pred.dropna()
    if not np.all(np.isin(y_pred.index, y_test.index)):
        raise IndexError("Predicted time points are not in test set")
    y_test = y_test.loc[y_pred.index]

    y_test[y_test == 0.0] = np.nan  # avoid divided zero
    return ((y_test - y_pred) / y_test).abs().mean()
