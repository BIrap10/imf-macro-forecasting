"""
arima_utils.py
Shared helper for forecast.py and anomaly_detection.py: instead of guessing
ARIMA(1,1,1) for every series, tests a handful of reasonable candidate orders
and picks whichever gets the lowest AIC (Akaike Information Criterion) - a
standard statistic that rewards a model for fitting well while penalizing it
for being needlessly complex. Lower AIC = better trade-off between fit and
simplicity.

This doesn't guarantee a better forecast (a well-chosen order can still lose
to a naive baseline, if the underlying series is close to a random walk - see
model_accuracy.py). What it does guarantee is that the order used is a
data-driven choice per series, not a single guess applied everywhere.
"""

import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Small, deliberately limited candidate set - enough to cover the common
# cases (AR-only, MA-only, differenced/undifferenced, mixed) without being
# so large that fitting every candidate for every series gets slow.
CANDIDATE_ORDERS = [
    (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1),
    (1, 1, 1), (2, 1, 0), (2, 1, 1), (1, 1, 2),
]


def select_arima_order(values, candidates=None):
    """Fit each candidate order, return the one with lowest AIC.
    Falls back to (1,1,1) if every candidate fails to fit (e.g. too little data)."""
    candidates = candidates or CANDIDATE_ORDERS
    best_order, best_aic, found_any = (1, 1, 1), np.inf, False
    for order in candidates:
        try:
            fitted = ARIMA(values, order=order).fit()
            if fitted.aic < best_aic:
                best_aic, best_order, found_any = fitted.aic, order, True
        except Exception:
            continue
    return best_order if found_any else (1, 1, 1)
