"""
generate_briefing.py
Reads forecasts.csv and anomalies.csv and asks Claude to draft a short,
readable macro briefing per country - the "agentic" layer that turns model
output into something a person would actually want to read.

Requires an Anthropic API key: https://console.anthropic.com/settings/keys
Set it as an environment variable before running:
    export ANTHROPIC_API_KEY="sk-ant-..."

Input:  data/forecasts.csv, data/anomalies.csv
Output: data/briefing.md
"""

import os
import pandas as pd
import requests

FORECASTS_PATH = "data/forecasts.csv"
ANOMALIES_PATH = "data/anomalies.csv"
OUTPUT_PATH = "data/briefing.md"
MODEL = "claude-sonnet-4-6"

INDICATOR_LABELS = {
    "real_gdp_growth_pct": "Real GDP growth (%)",
    "inflation_pct": "Inflation (%)",
    "fiscal_balance_pct_gdp": "Fiscal balance (% GDP)",
    "current_account_pct_gdp": "Current account (% GDP)",
    "gov_debt_pct_gdp": "Government debt (% GDP)",
}


def build_country_summary(country: str, forecasts: pd.DataFrame, anomalies: pd.DataFrame) -> str:
    """Build a compact plain-text data summary for one country to hand to the model."""
    cdf = forecasts[forecasts["country"] == country]
    lines = [f"Country: {country}"]

    actuals = cdf[cdf["value_type"] == "actual"]
    if not actuals.empty:
        latest_year = actuals["year"].max()
        lines.append(f"\nMost recent actual data ({latest_year}):")
        for ind, label in INDICATOR_LABELS.items():
            row = actuals[(actuals["indicator"] == ind) & (actuals["year"] == latest_year)]
            if not row.empty:
                lines.append(f"  {label}: {row['value'].iloc[0]:.1f}")

    fc = cdf[cdf["value_type"] == "arima_forecast"].sort_values("year")
    if not fc.empty:
        lines.append(f"\nARIMA forecast, next {fc['year'].nunique()} years:")
        for ind, label in INDICATOR_LABELS.items():
            series = fc[fc["indicator"] == ind].sort_values("year")
            if not series.empty:
                vals = ", ".join(f"{int(y)}: {v:.1f}" for y, v in zip(series["year"], series["value"]))
                lines.append(f"  {label}: {vals}")

    cadf = anomalies[(anomalies["country"] == country) & (anomalies["flagged"])]
    if not cadf.empty:
        top = cadf.reindex(cadf["z_score"].abs().sort_values(ascending=False).index).head(5)
        lines.append("\nNotable historical deviations from trend (backtest flags):")
        for _, row in top.iterrows():
            label = INDICATOR_LABELS.get(row["indicator"], row["indicator"])
            lines.append(f"  {int(row['year'])} {label}: actual {row['actual']:.1f} "
                        f"vs. expected {row['forecast']:.1f}")

    return "\n".join(lines)


def call_claude(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY found. Get one at "
            "https://console.anthropic.com/settings/keys and run:\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(block["text"] for block in data["content"] if block["type"] == "text")


def main():
    forecasts = pd.read_csv(FORECASTS_PATH)
    anomalies = pd.read_csv(ANOMALIES_PATH)

    countries = sorted(forecasts["country"].unique())
    sections = []

    for country in countries:
        summary = build_country_summary(country, forecasts, anomalies)
        prompt = (
            "You are a macroeconomic analyst drafting a short briefing note. "
            "Given the data below for one country, write 3-4 sentences in plain "
            "English covering: (1) where the economy stands now, (2) where the "
            "forecast points over the next few years, and (3) whether any "
            "historical deviations from trend are worth keeping in mind. "
            "Be specific with numbers, avoid hedging filler, and interpret the "
            "data rather than repeating it verbatim.\n\n" + summary
        )
        print(f"Drafting briefing for {country}...")
        text = call_claude(prompt)
        sections.append(f"## {country}\n\n{text}\n")

    output = "# Macro Briefing\n\n" + "\n".join(sections)
    with open(OUTPUT_PATH, "w") as f:
        f.write(output)
    print(f"Saved briefing to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
