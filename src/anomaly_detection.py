"""
anomaly_detection.py
Rolling-origin backtest: at each year, fits a simple ARIMA model on everything
before it, forecasts one step ahead, and compares that forecast to what actually
happened (per the WEO dataset). Years where the deviation is large relative to
that series' typical error get flagged.

Also computes a naive baseline forecast at each step (just carry forward last
year's value, unchanged) - this answers the question every forecast should be
able to answer: is the model actually adding value, or would a dead-simple
guess do just as well?

The ARIMA order is selected once per series (by AIC, via arima_utils.py),
using only the initial training window - not the full series - so the order
choice can't "see" the years it's about to be tested against. That avoids
leaking future information into the backtest.

Note: WEO data blends true historical actuals with the IMF's own forward
projections, with no flag distinguishing the two. So a "flagged" year means
"this diverged notably from a simple trend model" - which could reflect a real
economic shock, or the IMF revising its own outlook. Both are worth a second
look, which is the point of this module.

Input:  data/weo_macro_indicators.csv
Output: data/anomalies.csv
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arima_utils import select_arima_order

warnings.filterwarnings("ignore")

INPUT_PATH = "data/weo_macro_indicators.csv"
OUTPUT_PATH = "data/anomalies.csv"
MIN_HISTORY = 8           # minimum years of data before backtesting starts
FLAG_THRESHOLD_STD = 2.0  # flag if |residual| exceeds this many std devs

INDICATOR_COLS = [
    "real_gdp_growth_pct",
    "inflation_pct",
    "fiscal_balance_pct_gdp",
    "current_account_pct_gdp",
    "gov_debt_pct_gdp",
]


def one_step_residuals(series: pd.Series) -> pd.DataFrame:
    """Walk forward through the series: fit on everything before year t, forecast
    year t with both ARIMA and a naive (persistence) baseline, record both
    residuals. One row per testable year."""
    years = series.index.tolist()

    # Select order once, from the initial window only - avoids picking an
    # order based on data from years we're about to "forecast" in the loop.
    order = select_arima_order(series.iloc[:MIN_HISTORY].values)

    rows = []
    for i in range(MIN_HISTORY, len(years)):
        train = series.iloc[:i]
        actual = series.iloc[i]
        year = years[i]

        naive_forecast = float(train.iloc[-1])  # "predict no change from last year"

        try:
            fitted = ARIMA(train.values, order=order).fit()
            forecast = float(fitted.forecast(steps=1)[0])
            rows.append({"year": year, "actual": actual, "forecast": forecast,
                        "residual": actual - forecast,
                        "naive_forecast": naive_forecast,
                        "naive_residual": actual - naive_forecast,
                        "arima_order": str(order)})
        except Exception:
            continue
    return pd.DataFrame(rows)


def run(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path).sort_values(["country", "year"])
    all_flags = []

    for country, cdf in df.groupby("country"):
        cdf = cdf.set_index("year").sort_index()
        for ind in INDICATOR_COLS:
            series = cdf[ind].dropna()
            if len(series) < MIN_HISTORY + 3:
                continue
            resid_df = one_step_residuals(series)
            if resid_df.empty:
                continue
            std = resid_df["residual"].std()
            if not std or np.isnan(std):
                continue
            resid_df["z_score"] = resid_df["residual"] / std
            resid_df["flagged"] = resid_df["z_score"].abs() > FLAG_THRESHOLD_STD
            resid_df["country"] = country
            resid_df["indicator"] = ind
            all_flags.append(resid_df)

    result = pd.concat(all_flags, ignore_index=True)
    result = result[["country", "indicator", "year", "actual", "forecast",
                     "residual", "naive_forecast", "naive_residual",
                     "z_score", "flagged", "arima_order"]]
    result = result.sort_values(["country", "indicator", "year"])
    result.to_csv(output_path, index=False)

    n_flagged = int(result["flagged"].sum())
    print(f"Saved {len(result)} rows to {output_path} ({n_flagged} flagged as anomalies)")

    # Quick ARIMA-vs-naive scoreboard, printed here so it's visible on every run
    arima_mae = result["residual"].abs().mean()
    naive_mae = result["naive_residual"].abs().mean()
    verdict = "beats" if arima_mae < naive_mae else "loses to"
    print(f"\nARIMA MAE: {arima_mae:.2f}  |  Naive baseline MAE: {naive_mae:.2f}  "
          f"-> ARIMA {verdict} the naive baseline overall")

    orders_used = (result[["country", "indicator", "arima_order"]]
                  .drop_duplicates()
                  .sort_values(["country", "indicator"]))
    print(f"\nSelected ARIMA orders (by AIC, {orders_used['arima_order'].nunique()} distinct orders used):")
    print(orders_used.to_string(index=False))

    if n_flagged:
        top = result[result["flagged"]].reindex(
            result[result["flagged"]]["z_score"].abs().sort_values(ascending=False).index
        )
        print("\nTop flagged deviations:")
        print(top.head(10).to_string(index=False))
    return result


if __name__ == "__main__":
    run()
