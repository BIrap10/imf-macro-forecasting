# IMF Macro Forecasting Assistant

An agentic pipeline that pulls IMF World Economic Outlook (WEO) macro data,
generates statistical forecasts, flags meaningful deviations from trend, and
uses an LLM to draft a plain-English briefing on the results.

## Why this project

Built to develop hands-on macroeconomic modeling and forecasting skills using
real IMF data, and to explore how agentic AI tools can support macro analysis
and reporting workflows.

## Pipeline

1. **Data** — pull WEO indicators (real GDP growth, inflation, fiscal balance,
   current account, government debt) via the public IMF DataMapper API.
2. **Forecast** — fit ARIMA/VAR models per country and indicator.
3. **Anomaly flagging** — flag cases where actuals deviate meaningfully from
   forecast/trend.
4. **Agent briefing** — an LLM takes the forecast output and flags and drafts
   a short, readable summary of what changed and why it matters.

## Status

- [x] Data pipeline — `src/fetch_weo_data.py`
- [x] Forecasting module — `src/forecast.py` (ARIMA + VAR)
- [x] Anomaly detection — `src/anomaly_detection.py` (rolling-origin backtest)
- [ ] LLM briefing agent

## Project structure
## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/fetch_weo_data.py   # pulls WEO data -> data/weo_macro_indicators.csv
python src/forecast.py         # generates forecasts -> data/forecasts.csv
python src/anomaly_detection.py  # flags deviations -> data/anomalies.csv
```

`forecast.py` fits two models per country:
- **ARIMA(1,1,1)** — one model per indicator, univariate
- **VAR** — one model per country, jointly fitting all five indicators to
  capture cross-variable dynamics (e.g. how inflation and fiscal balance move
  together)

Both forecast types are written to `data/forecasts.csv` alongside the actuals,
tagged by `value_type`, so they can be compared directly.

`anomaly_detection.py` runs a rolling-origin backtest: at each year, it fits a
model on everything before it, forecasts one step ahead, and flags years where
the actual value deviates more than 2 standard deviations from what the model
expected. Note that WEO data blends true historical actuals with the IMF's own
forward projections with no flag distinguishing the two, so a flagged year
means "this diverged notably from a simple trend model" - worth a second look,
whether that's a real economic shock or a revised outlook.

## Data source

[IMF DataMapper API](https://www.imf.org/external/datamapper/api/help) —
public, no API key required. Indicators and country coverage per the IMF
World Economic Outlook database.

## License

MIT
