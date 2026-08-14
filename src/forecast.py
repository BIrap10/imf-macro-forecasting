"""
forecast.py
Forecasts IMF WEO macro indicators using two approaches:
  - ARIMA: univariate, one model per (country, indicator)
  - VAR:   multivariate, one model per country, jointly modeling all
           indicators to capture cross-variable dynamics (e.g. how inflation
           and fiscal balance move together)

Input:  data/weo_macro_indicators.csv (wide format, produced by fetch_weo_data.py)
Output: data/forecasts.csv (long format: country, indicator, year, value,
        value_type ["actual"|"arima_forecast"|"var_forecast"], lower_95, upper_95)

Both ARIMA and VAR forecasts include 95% confidence intervals - a range
("likely between X and Y"), not just a single point number. Actual rows have
no interval (there's nothing uncertain about confirmed history).
"""

import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from arima_utils import select_arima_order

warnings.filterwarnings("ignore")  # statsmodels convergence warnings are noisy on short series

INPUT_PATH = "data/weo_macro_indicators.csv"
OUTPUT_PATH = "data/forecasts.csv"
FORECAST_YEARS = 3
INDICATOR_COLS = [
    "real_gdp_growth_pct",
    "inflation_pct",
    "fiscal_balance_pct_gdp",
    "current_account_pct_gdp",
    "gov_debt_pct_gdp",
]


def forecast_arima(series: pd.Series, steps: int) -> pd.DataFrame:
    """Pick the best-fitting ARIMA order for this series (by AIC), fit it,
    and forecast ahead with a 95% confidence interval."""
    order = select_arima_order(series.values)
    fitted = ARIMA(series.values, order=order).fit()
    forecast_result = fitted.get_forecast(steps=steps)
    summary = forecast_result.summary_frame(alpha=0.05)  # 95% CI
    return summary[["mean", "mean_ci_lower", "mean_ci_upper"]]


def forecast_var(country_df: pd.DataFrame, steps: int):
    """Fit a VAR model jointly on all indicators for one country and forecast
    ahead, with a 95% confidence interval for each indicator."""
    values = country_df[INDICATOR_COLS].dropna()
    lag_order = 1 if len(values) < 15 else 2  # keep lag order sane for short series
    model = VAR(values.values)
    fitted = model.fit(lag_order)
    point, lower, upper = fitted.forecast_interval(values.values[-lag_order:],
                                                    steps=steps, alpha=0.05)
    return (pd.DataFrame(point, columns=INDICATOR_COLS),
            pd.DataFrame(lower, columns=INDICATOR_COLS),
            pd.DataFrame(upper, columns=INDICATOR_COLS))


def run(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path).sort_values(["country", "year"])
    all_rows = []

    for country, cdf in df.groupby("country"):
        cdf = cdf.sort_values("year")
        last_year = int(cdf["year"].max())
        future_years = list(range(last_year + 1, last_year + 1 + FORECAST_YEARS))

        # Record actuals - no interval, since there's nothing uncertain about
        # confirmed history
        for _, row in cdf.iterrows():
            for ind in INDICATOR_COLS:
                if pd.notna(row[ind]):
                    all_rows.append({"country": country, "indicator": ind,
                                     "year": int(row["year"]), "value": row[ind],
                                     "value_type": "actual",
                                     "lower_95": None, "upper_95": None})

        # ARIMA — per indicator, with confidence interval
        for ind in INDICATOR_COLS:
            series = cdf[ind].dropna()
            if len(series) < 8:
                continue  # not enough history for a stable fit
            try:
                preds = forecast_arima(series, FORECAST_YEARS)
                for yr, row in zip(future_years, preds.itertuples()):
                    all_rows.append({"country": country, "indicator": ind, "year": yr,
                                     "value": row.mean, "value_type": "arima_forecast",
                                     "lower_95": row.mean_ci_lower,
                                     "upper_95": row.mean_ci_upper})
            except Exception as e:
                print(f"  ARIMA skipped for {country}/{ind}: {e}")

        # VAR — jointly across indicators, with confidence interval
        if cdf[INDICATOR_COLS].dropna().shape[0] >= 8:
            try:
                point, lower, upper = forecast_var(cdf, FORECAST_YEARS)
                for i, yr in enumerate(future_years):
                    for ind in INDICATOR_COLS:
                        all_rows.append({"country": country, "indicator": ind, "year": yr,
                                         "value": point.iloc[i][ind],
                                         "value_type": "var_forecast",
                                         "lower_95": lower.iloc[i][ind],
                                         "upper_95": upper.iloc[i][ind]})
            except Exception as e:
                print(f"  VAR skipped for {country}: {e}")

    result = pd.DataFrame(all_rows).sort_values(["country", "indicator", "year", "value_type"])
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")
    return result


if __name__ == "__main__":
    run()
