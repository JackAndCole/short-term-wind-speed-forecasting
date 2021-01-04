import time

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import check_cv
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection._tune import _fit_and_score
from sktime.utils.validation.forecasting import check_scoring
from sktime.utils.validation.forecasting import check_y


class ParallelForecastingGridSearchCV(ForecastingGridSearchCV):

    def fit(self, y_train, fh=None, X_train=None, **fit_params):
        """Parallel fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        y_train = check_y(y_train)

        # validate cross-validator
        cv = check_cv(self.cv)
        base_forecaster = clone(self.forecaster)

        scoring = check_scoring(self.scoring)
        scorers = {scoring.name: scoring}
        refit_metric = scoring.name

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose
        )

        results = {}
        all_candidate_params = []
        all_out = []

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                n_splits = cv.get_n_splits(y_train)
                print("Fitting {0} folds for each of {1} candidates,"
                      " totalling {2} fits".format(n_splits, n_candidates,
                                                   n_candidates * n_splits))

            parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
            out = parallel(delayed(_fit_and_score)(clone(base_forecaster),
                                                   cv,
                                                   y_train,
                                                   X_train,
                                                   parameters=parameters,
                                                   **fit_and_score_kwargs) for parameters in candidate_params)

            n_splits = cv.get_n_splits(y_train)

            if len(out) < 1:
                raise ValueError("No fits were performed. "
                                 "Was the CV iterator empty? "
                                 "Were there no candidates?")

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            nonlocal results
            results = self._format_results(
                all_candidate_params, scorers, all_out)
            return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_%s"
                                   % refit_metric].argmin()
        self.best_score_ = results["mean_test_%s" % refit_metric][
            self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.best_forecaster_ = clone(base_forecaster).set_params(
            **self.best_params_)

        if self.refit:
            refit_start_time = time.time()
            self.best_forecaster_.fit(y_train, fh=fh, X_train=X_train,
                                      **fit_params)
            self.refit_time_ = time.time() - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers[scoring.name]

        self.cv_results_ = results
        self.n_splits_ = cv.get_n_splits(y_train)

        self._is_fitted = True
        return self
