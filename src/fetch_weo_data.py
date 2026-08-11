"""
fetch_weo_data.py
Pulls IMF World Economic Outlook (WEO) indicators via the public DataMapper API
(no API key required) and saves them to a tidy CSV for downstream forecasting.

API docs: https://www.imf.org/external/datamapper/api/help
"""

import os
import requests
import pandas as pd

BASE_URL = "https://www.imf.org/external/datamapper/api/v1"

# Core macro indicators for a forecasting/agent project
INDICATORS = {
    "NGDP_RPCH": "real_gdp_growth_pct",
    "PCPIPCH": "inflation_pct",
    "GGXCNL_NGDP": "fiscal_balance_pct_gdp",
    "BCA_NGDPD": "current_account_pct_gdp",
    "GGXWDG_NGDP": "gov_debt_pct_gdp",
}

# Pick your countries (ISO alpha-3 codes)
COUNTRIES = ["USA", "GBR", "DEU", "JPN", "BRA"]

OUTPUT_PATH = "data/weo_macro_indicators.csv"

HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "*",
    "Accept-Language": "*",
}


def fetch_indicator(indicator_code: str, countries: list[str]) -> pd.DataFrame:
    """Fetch one indicator, returning only the requested countries.

    Note: the DataMapper API doesn't reliably filter by country when multiple
    country codes are chained in the URL path - it can return every country
    it has data for. So we fetch the indicator and filter client-side instead
    of trusting the URL to do it.
    """
    url = f"{BASE_URL}/{indicator_code}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()["values"][indicator_code]

    rows = []
    for country, series in payload.items():
        if country not in countries:
            continue
        for year, value in series.items():
            rows.append({"country": country, "year": int(year),
                        "indicator": indicator_code, "value": value})
    return pd.DataFrame(rows)


def main():
    frames = [fetch_indicator(code, COUNTRIES) for code in INDICATORS]
    df = pd.concat(frames, ignore_index=True)
    df["indicator"] = df["indicator"].map(INDICATORS)  # human-readable names

    # Long -> wide: one row per country-year, one column per indicator
    wide = df.pivot_table(index=["country", "year"], columns="indicator",
                          values="value").reset_index()
    wide = wide.sort_values(["country", "year"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    wide.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(wide)} rows to {OUTPUT_PATH}")
    print(wide.tail(10))


if __name__ == "__main__":
    main()
