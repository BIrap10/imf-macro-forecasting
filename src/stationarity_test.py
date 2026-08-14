"""
stationarity_test.py
Runs the Augmented Dickey-Fuller (ADF) test - the standard first diagnostic
step in any real time-series workflow - on every (country, indicator) series,
both at the raw level and after one differencing pass.

The ADF test's null hypothesis is that a series has a unit root (i.e. is NOT
stationary - it wanders rather than reverting to a stable mean/trend). A low
p-value (< 0.05) lets us reject that null and conclude the series IS
stationary at that significance level.

This cross-checks arima_utils.py's AIC-based order selection: if a raw series
is found non-stationary but its once-differenced version is, that confirms
d=1 was the statistically correct call - not just an AIC-preferred guess made
without a formal test behind it.

Input:  data/weo_macro_indicators.csv
Output: data/stationarity_report.csv, printed summary
"""

import warnings
import pandas as pd
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

INPUT_PATH = "data/weo_macro_indicators.csv"
OUTPUT_PATH = "data/stationarity_report.csv"
SIGNIFICANCE = 0.05

INDICATOR_COLS = [
    "real_gdp_growth_pct",
    "inflation_pct",
    "fiscal_balance_pct_gdp",
    "current_account_pct_gdp",
    "gov_debt_pct_gdp",
]

INDICATOR_LABELS = {
    "real_gdp_growth_pct": "Real GDP growth (%)",
    "inflation_pct": "Inflation (%)",
    "fiscal_balance_pct_gdp": "Fiscal balance (% GDP)",
    "current_account_pct_gdp": "Current account (% GDP)",
    "gov_debt_pct_gdp": "Government debt (% GDP)",
}


def adf_result(series):
    """Run ADF test, return (statistic, p_value, is_stationary_at_5pct)."""
    try:
        stat, pvalue, *_ = adfuller(series, autolag="AIC")
        return stat, pvalue, pvalue < SIGNIFICANCE
    except Exception:
        return None, None, None


def run(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path).sort_values(["country", "year"])
    rows = []

    for country, cdf in df.groupby("country"):
        cdf = cdf.set_index("year").sort_index()
        for ind in INDICATOR_COLS:
            series = cdf[ind].dropna()
            if len(series) < 10:
                continue

            level_stat, level_p, level_stationary = adf_result(series)
            diffed = series.diff().dropna()
            diff_stat, diff_p, diff_stationary = adf_result(diffed)

            recommended_d = 0 if level_stationary else (1 if diff_stationary else 2)
            rows.append({
                "country": country, "indicator": ind,
                "level_adf_stat": level_stat, "level_p_value": level_p,
                "level_stationary": level_stationary,
                "diff_adf_stat": diff_stat, "diff_p_value": diff_p,
                "diff_stationary": diff_stationary,
                "recommended_d": recommended_d,
            })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    n_level_stationary = int(result["level_stationary"].sum())
    n_total = len(result)
    print(f"Saved {n_total} rows to {output_path}")
    print(f"\n{n_level_stationary} of {n_total} series are stationary at the raw level "
          f"(p < {SIGNIFICANCE}). {n_total - n_level_stationary} required differencing.")

    print("\nRecommended differencing order (d), by count:")
    print(result["recommended_d"].value_counts().sort_index().to_string())

    print("\nBy indicator (share stationary at raw level):")
    summary = result.groupby("indicator").agg(
        n=("level_stationary", "count"),
        pct_stationary_at_level=("level_stationary", "mean"),
    ).reset_index()
    summary["label"] = summary["indicator"].map(INDICATOR_LABELS)
    summary["pct_stationary_at_level"] = (summary["pct_stationary_at_level"] * 100).round(0)
    print(summary[["label", "n", "pct_stationary_at_level"]].to_string(index=False))

    still_bad = result[(~result["diff_stationary"].fillna(False)) &
                       (~result["level_stationary"].fillna(False))]
    if len(still_bad):
        print(f"\n{len(still_bad)} series still non-stationary even after differencing once "
              f"- worth a second look:")
        print(still_bad[["country", "indicator"]].to_string(index=False))

    return result


if __name__ == "__main__":
    run()
