"""
model_accuracy.py
Turns "the model seems reasonable" into a measured claim. Computes standard
accuracy metrics (MAE, RMSE) from two different sources:

  1. Historical backtest accuracy - from anomaly_detection.py's rolling-origin
     test, where "actual" is confirmed history. This is genuine forecast
     accuracy, measured against reality.
  2. IMF agreement - from compare_to_imf.py's 2024-2026 hold-out, measuring
     how closely our model tracks the IMF's own current published numbers.
     Framed as "agreement," not "accuracy," since not all of 2024-2026 is
     necessarily confirmed outcome vs. IMF's own projection.

MAE (Mean Absolute Error): average size of the error, in the indicator's own
units (e.g. "off by 0.6 percentage points on average"). Easy to interpret.
RMSE (Root Mean Squared Error): similar, but penalizes large misses more
heavily. If RMSE is much bigger than MAE, it means a few big outliers are
driving the error - which lines up with what anomaly_detection.py flags.

Also reports ARIMA vs. a naive baseline (predict "no change from last year")
per indicator - the standard sanity check every forecasting model should pass:
is it actually beating the simplest possible guess?

Input:  data/anomalies.csv, data/imf_comparison.csv
Output: data/accuracy_report.csv, printed summary
"""

import pandas as pd
import numpy as np

BACKTEST_PATH = "data/anomalies.csv"
COMPARISON_PATH = "data/imf_comparison.csv"
OUTPUT_PATH = "data/accuracy_report.csv"

INDICATOR_LABELS = {
    "real_gdp_growth_pct": "Real GDP growth (%)",
    "inflation_pct": "Inflation (%)",
    "fiscal_balance_pct_gdp": "Fiscal balance (% GDP)",
    "current_account_pct_gdp": "Current account (% GDP)",
    "gov_debt_pct_gdp": "Government debt (% GDP)",
}


def compute_metrics(df: pd.DataFrame, error_col: str, group_cols: list) -> pd.DataFrame:
    """MAE and RMSE of error_col, optionally grouped."""
    if group_cols:
        grouped = df.groupby(group_cols).agg(
            n=(error_col, "count"),
            mae=(error_col, lambda x: x.abs().mean()),
            rmse=(error_col, lambda x: np.sqrt((x ** 2).mean())),
        ).reset_index()
    else:
        grouped = pd.DataFrame([{
            "n": df[error_col].count(),
            "mae": df[error_col].abs().mean(),
            "rmse": np.sqrt((df[error_col] ** 2).mean()),
        }])
    return grouped


def run(backtest_path: str = BACKTEST_PATH, comparison_path: str = COMPARISON_PATH,
        output_path: str = OUTPUT_PATH):
    backtest = pd.read_csv(backtest_path)
    comparison = pd.read_csv(comparison_path)

    # --- Historical backtest accuracy (measured against confirmed reality) ---
    bt_overall = compute_metrics(backtest, "residual", [])
    bt_by_indicator = compute_metrics(backtest, "residual", ["indicator"])
    bt_by_indicator["label"] = bt_by_indicator["indicator"].map(INDICATOR_LABELS)

    print("=" * 70)
    print("HISTORICAL BACKTEST ACCURACY (1-step-ahead ARIMA, vs. confirmed history)")
    print("=" * 70)
    print(f"{int(bt_overall['n'].iloc[0])} country-year forecasts tested across "
          f"{backtest['country'].nunique()} countries, {backtest['indicator'].nunique()} indicators.")
    print(f"Overall MAE: {bt_overall['mae'].iloc[0]:.2f}   "
          f"Overall RMSE: {bt_overall['rmse'].iloc[0]:.2f}\n")
    print("By indicator:")
    print(bt_by_indicator[["label", "n", "mae", "rmse"]]
          .sort_values("mae")
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # --- ARIMA vs. naive baseline: is the model actually adding value? ---
    naive_by_indicator = compute_metrics(backtest, "naive_residual", ["indicator"])
    naive_by_indicator = naive_by_indicator.rename(columns={"mae": "naive_mae", "rmse": "naive_rmse"})
    scoreboard = bt_by_indicator.merge(naive_by_indicator[["indicator", "naive_mae", "naive_rmse"]],
                                       on="indicator")
    scoreboard["arima_wins"] = scoreboard["mae"] < scoreboard["naive_mae"]
    scoreboard["improvement_pct"] = (
        (scoreboard["naive_mae"] - scoreboard["mae"]) / scoreboard["naive_mae"] * 100
    )

    print("\n" + "-" * 70)
    print("ARIMA vs. NAIVE BASELINE (predict 'no change from last year')")
    print("-" * 70)
    for _, row in scoreboard.sort_values("improvement_pct", ascending=False).iterrows():
        verdict = "beats" if row["arima_wins"] else "LOSES TO"
        print(f"  {row['label']:<28} ARIMA MAE {row['mae']:.2f}  vs  "
              f"Naive MAE {row['naive_mae']:.2f}  -> ARIMA {verdict} naive "
              f"({row['improvement_pct']:+.0f}%)")
    n_wins = int(scoreboard["arima_wins"].sum())
    print(f"\nARIMA beats the naive baseline on {n_wins} of {len(scoreboard)} indicators.")

    # --- Agreement with IMF's current published numbers, 2024-2026 ---
    cmp_overall = compute_metrics(comparison, "difference", [])
    cmp_by_indicator = compute_metrics(comparison, "difference", ["indicator"])
    cmp_by_indicator["label"] = cmp_by_indicator["indicator"].map(INDICATOR_LABELS)

    print("\n" + "=" * 70)
    print("AGREEMENT WITH IMF's CURRENT OUTLOOK (2024-2026, our forecast vs. IMF)")
    print("=" * 70)
    print(f"Overall MAE: {cmp_overall['mae'].iloc[0]:.2f}   "
          f"Overall RMSE: {cmp_overall['rmse'].iloc[0]:.2f}\n")
    print("By indicator:")
    print(cmp_by_indicator[["label", "n", "mae", "rmse"]]
          .sort_values("mae")
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Save full breakdown, both sources, clearly labeled
    bt_by_indicator["source"] = "historical_backtest"
    cmp_by_indicator["source"] = "imf_agreement_2024_2026"
    combined = pd.concat([
        bt_by_indicator[["source", "indicator", "label", "n", "mae", "rmse"]],
        cmp_by_indicator[["source", "indicator", "label", "n", "mae", "rmse"]],
    ], ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSaved full breakdown to {output_path}")


if __name__ == "__main__":
    run()
