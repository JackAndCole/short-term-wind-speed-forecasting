import pandas as pd
from sklearn.base import clone


def multistep_forecasting(forecaster, y_train, y_test, steps):
    """multi-step forecasting."""

    def forecasting(base_forecaster, fh):
        if hasattr(base_forecaster, "cv"):
            base_forecaster.cv.fh = fh

        base_forecaster.fit(y_train, fh=fh)
        return base_forecaster.update_predict(y_test)

    y_preds = pd.concat([forecasting(clone(forecaster), [step]) for step in steps], axis=1)
    y_preds.columns = ["step%d" % i for i in steps]

    return y_preds
