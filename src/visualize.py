"""
visualize.py
Generates three charts from the project's existing outputs:
  1. Forecast fan     - actual history + ARIMA/VAR forecast, one panel per country
  2. IMF comparison   - heatmap of where our model agrees/disagrees with IMF's
                         current outlook, country x indicator
  3. Top anomalies    - the largest flagged deviations from anomaly_detection.py

Input:  data/forecasts.csv, data/imf_comparison.csv, data/anomalies.csv
Output: charts/forecast_fan.png, charts/imf_comparison_heatmap.png,
        charts/top_anomalies.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FORECASTS_PATH = "data/forecasts.csv"
COMPARISON_PATH = "data/imf_comparison.csv"
ANOMALIES_PATH = "data/anomalies.csv"
CHARTS_DIR = "charts"

INDICATOR_LABELS = {
    "real_gdp_growth_pct": "Real GDP growth (%)",
    "inflation_pct": "Inflation (%)",
    "fiscal_balance_pct_gdp": "Fiscal balance (% GDP)",
    "current_account_pct_gdp": "Current account (% GDP)",
    "gov_debt_pct_gdp": "Government debt (% GDP)",
}


def plot_forecast_fan(indicator: str = "real_gdp_growth_pct"):
    """Small-multiples chart: actual history + ARIMA/VAR forecast, per country."""
    df = pd.read_csv(FORECASTS_PATH)
    df = df[df["indicator"] == indicator]
    countries = sorted(df["country"].unique())

    ncols = 4
    nrows = -(-len(countries) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=False)
    axes = axes.flatten()

    for i, country in enumerate(countries):
        ax = axes[i]
        cdf = df[df["country"] == country]

        actual = cdf[cdf["value_type"] == "actual"].sort_values("year")
        arima = cdf[cdf["value_type"] == "arima_forecast"].sort_values("year")
        var = cdf[cdf["value_type"] == "var_forecast"].sort_values("year")

        ax.plot(actual["year"], actual["value"], color="#16324F", linewidth=1.5, label="Actual")
        if not arima.empty:
            bridge_y = list(actual["value"].tail(1)) + list(arima["value"])
            bridge_x = list(actual["year"].tail(1)) + list(arima["year"])
            ax.plot(bridge_x, bridge_y, color="#C0392B", linestyle="--", linewidth=1.3, label="ARIMA")
        if not var.empty:
            bridge_y = list(actual["value"].tail(1)) + list(var["value"])
            bridge_x = list(actual["year"].tail(1)) + list(var["year"])
            ax.plot(bridge_x, bridge_y, color="#27AE60", linestyle=":", linewidth=1.5, label="VAR")

        ax.set_title(country, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    for j in range(len(countries), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(f"{INDICATOR_LABELS.get(indicator, indicator)} — actual vs. forecast",
                fontsize=13, y=1.06)
    fig.tight_layout()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = f"{CHARTS_DIR}/forecast_fan.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_imf_comparison_heatmap():
    """Heatmap: country x indicator, colored by average (our_forecast - imf_projection)."""
    df = pd.read_csv(COMPARISON_PATH)
    pivot = df.groupby(["country", "indicator"])["difference"].mean().unstack()
    pivot = pivot[[c for c in INDICATOR_LABELS if c in pivot.columns]]  # consistent column order
    pivot.columns = [INDICATOR_LABELS[c] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(pivot) + 2))
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > vmax * 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Our forecast minus IMF projection (avg, 2024-2026)")
    ax.set_title("Where our model diverges from IMF's current outlook", fontsize=12, pad=12)
    fig.tight_layout()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = f"{CHARTS_DIR}/imf_comparison_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_top_anomalies(n: int = 15):
    """Horizontal bar chart of the largest flagged deviations from the backtest."""
    df = pd.read_csv(ANOMALIES_PATH)
    flagged = df[df["flagged"]].copy()
    flagged["abs_z"] = flagged["z_score"].abs()
    top = flagged.sort_values("abs_z", ascending=False).head(n)
    top = top.iloc[::-1]  # largest at top of chart

    labels = [f"{row.country} {INDICATOR_LABELS.get(row.indicator, row.indicator)} {int(row.year)}"
              for row in top.itertuples()]
    colors = ["#C0392B" if z < 0 else "#27AE60" for z in top["z_score"]]

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(top) + 1.5))
    ax.barh(labels, top["z_score"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Z-score (std devs actual deviated from expected)")
    ax.set_title("Largest flagged deviations from trend (rolling-origin backtest)", fontsize=12)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = f"{CHARTS_DIR}/top_anomalies.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    plot_forecast_fan("real_gdp_growth_pct")
    plot_imf_comparison_heatmap()
    plot_top_anomalies()
