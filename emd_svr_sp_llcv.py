import os

import numpy as np
import pandas as pd
from PyEMD import EMD
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sktime.forecasting.compose import ReducedRegressionForecaster
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.performance_metrics.forecasting import make_forecasting_scorer
from sktime.utils.plotting.forecasting import plot_ys

from config import DefaultConfig
from sktime_fix.model_selection import ParallelForecastingGridSearchCV
from utils import load_data, multistep_forecasting, root_mean_squared_error


def main(opt, verbose=0):
    wind_speed = load_data(station=opt.station)

    y_train, y_test = wind_speed.iloc[:-opt.test_size], wind_speed.iloc[-opt.test_size:]
    plot_ys(y_train, y_test, labels=("y_train", "y_test"))

    # ================================== Model ==================================

    emd = EMD()
    imfs = emd(wind_speed.values).T

    num_imfs = imfs.shape[1]
    imfs = pd.DataFrame(imfs, index=pd.RangeIndex(start=0, stop=len(imfs), step=1),
                        columns=["imf%d" % i for i in range(num_imfs - 1)] + ["residue"])

    y_trains_, y_tests_ = imfs.iloc[:-opt.test_size], imfs.iloc[-opt.test_size:]

    index = imfs.index[-opt.test_size:]
    columns = pd.MultiIndex.from_product([["imf%d" % i for i in range(num_imfs)], ["step%d" % i for i in opt.steps]])
    y_preds = pd.DataFrame(np.full((len(index), len(columns)), np.nan), index=index, columns=columns)

    for i in range(num_imfs):
        print("imf%d:" % i if i != num_imfs - 1 else "residue:")
        y_train_, y_test_ = y_trains_.iloc[:, i], y_tests_.iloc[:, i]

        if i in [0]:
            param_grid = {
                "regressor__clf__C": [1, 5, 10, 25, 50, 100, 150],
                "regressor__clf__gamma": ['scale', 0.001, 0.01, 0.1, 1.0],
                'regressor__fs__percentile': range(10, 100, 10),
            }
            regressor = Pipeline([
                ("fs", SelectPercentile(percentile=50, score_func=f_regression)),
                ("clf", SVR(C=5, gamma="scale"))
            ])
        else:
            param_grid = {
                "regressor__normalize": [True, False]
            }
            regressor = LassoLarsCV()
        forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=opt.window_length,
                                                 strategy=opt.strategy)
        grid_search = ParallelForecastingGridSearchCV(
            forecaster,
            cv=SlidingWindowSplitter(initial_window=int(len(y_train_) * 0.7)),
            param_grid=param_grid,
            scoring=make_forecasting_scorer(root_mean_squared_error, name="rmse"),
            n_jobs=opt.n_jobs,
            verbose=verbose
        )
        y_preds_ = multistep_forecasting(grid_search, y_train_, y_test_, steps=opt.steps)
        print([root_mean_squared_error(y_test_, y_preds_["step%d" % step]) for step in opt.steps])
        y_preds["imf%d" % i] = y_preds_

    y_preds = y_preds.swaplevel(1, 0, axis=1)
    y_preds = pd.concat([y_preds["step%d" % step].sum(axis=1, skipna=False) for step in opt.steps], axis=1)
    y_preds.columns = ["step%d" % i for i in opt.steps]
    y_preds.to_excel("output/%s_%s.xls" % (opt.station, os.path.split(__file__)[-1].rsplit(".")[0].upper()))

    print([root_mean_squared_error(y_test, y_preds["step%d" % step]) for step in opt.steps])


if __name__ == "__main__":
    opt = DefaultConfig()
    # opt.station = "S.W. PIER MI"
    main(opt)
