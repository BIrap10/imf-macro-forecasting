"""
compare_to_imf.py
Holds out recent years, forecasts them with our own ARIMA model using only
data from before the cutoff, and compares that forecast to what the IMF
itself currently projects for those same years (already present in the WEO
dataset). This answers a present-tense question - does our simple model
agree with IMF's own current outlook, or diverge from it - rather than the
historical backtest in anomaly_detection.py.

Input:  data/weo_macro_indicators.csv
Output: data/imf_comparison.csv
"""

import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

INPUT_PATH = "data/weo_macro_indicators.csv"
OUTPUT_PATH = "data/imf_comparison.csv"
CUTOFF_YEAR = 2023   # pretend we're standing at the end of this year
CHECK_YEARS = 3      # check this many years past the cutoff (e.g. 2024-2026)

INDICATOR_COLS = [
    "real_gdp_growth_pct",
    "inflation_pct",
    "fiscal_balance_pct_gdp",
    "current_account_pct_gdp",
    "gov_debt_pct_gdp",
]


def run(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path).sort_values(["country", "year"])
    rows = []

    for country, cdf in df.groupby("country"):
        cdf = cdf.set_index("year").sort_index()
        for ind in INDICATOR_COLS:
            series = cdf[ind].dropna()
            train = series[series.index <= CUTOFF_YEAR]
            check_years = [y for y in range(CUTOFF_YEAR + 1, CUTOFF_YEAR + 1 + CHECK_YEARS)
                          if y in series.index]
            if len(train) < 8 or not check_years:
                continue
            try:
                fitted = ARIMA(train.values, order=(1, 1, 1)).fit()
                forecast = fitted.forecast(steps=len(check_years))
            except Exception as e:
                print(f"  Skipped {country}/{ind}: {e}")
                continue
            for yr, our_val in zip(check_years, forecast):
                imf_val = series.loc[yr]
                rows.append({
                    "country": country, "indicator": ind, "year": yr,
                    "our_forecast": round(float(our_val), 2),
                    "imf_projection": round(float(imf_val), 2),
                    "difference": round(float(our_val - imf_val), 2),
                })

    result = pd.DataFrame(rows).sort_values(["country", "indicator", "year"])
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} rows to {output_path}")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
