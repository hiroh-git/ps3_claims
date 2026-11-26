import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    auc,
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
)


def evaluate_predictions(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike,
    tweedie_power: float = 1.5,
) -> pd.DataFrame:
    """Compute performance metrics of predictions against actual outcomes.

    Parameters
    ----------
    y_true : ArrayLike
        Actual outcomes (here: pure premium).
    y_pred : ArrayLike
        Model predictions.
    sample_weight : ArrayLike
        Weight of each observation (here: exposure).
    tweedie_power : float, optional
        Power of the Tweedie deviance, by default 1.5.

    Returns
    -------
    pd.DataFrame
        One column "value" with the metric names as index.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    w = np.asarray(sample_weight)

    bias = np.average(y_pred, weights=w) - np.average(y_true, weights=w)
    deviance = mean_tweedie_deviance(
        y_true, y_pred, sample_weight=w, power=tweedie_power
    )
    mae = mean_absolute_error(y_true, y_pred, sample_weight=w)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=w))

    # Gini as in the Lorenz curve of ps3_script: sort by prediction, then
    # accumulate the actual claim amounts (pure premium * exposure).
    ranking = np.argsort(y_pred)
    cum_claims = np.cumsum(y_true[ranking] * w[ranking])
    cum_claims = np.r_[0.0, cum_claims / cum_claims[-1]]
    cum_share = np.linspace(0, 1, len(cum_claims))
    gini = 1 - 2 * auc(cum_share, cum_claims)

    return pd.DataFrame(
        {"value": [bias, deviance, mae, rmse, gini]},
        index=["bias", "deviance", "mae", "rmse", "gini"],
    )
