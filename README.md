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
   current account, government debt) via the public IMF DataMapper API, for
   Italy, Greece, Spain, Canada, UK, China, and the US.
2. **Forecast** — fit ARIMA/VAR models per country and indicator, with ARIMA
   order chosen per series by AIC rather than a fixed guess, and 95%
   confidence intervals for every forecast rather than a bare point number.
3. **Anomaly flagging** — flag cases where actuals deviate meaningfully from
   forecast/trend (historical backtest).
4. **IMF comparison** — hold out recent years, forecast them independently,
   and compare to what the IMF itself currently projects for those same
   years - a present-tense check, not a historical one.
5. **Agent briefing** — an LLM takes the forecast output and flags and drafts
   a short, readable summary of what changed and why it matters.
6. **Visualization** — chart the forecasts, IMF comparison, and flagged
   anomalies for quick visual review.
7. **Accuracy metrics** — compute MAE/RMSE from both backtests, and check ARIMA
   against a naive baseline, turning "the model seems reasonable" into a
   measured claim.
8. **Stationarity testing** — formal ADF test on every series, cross-checking
   whether the differencing AIC selected actually matches a statistical test.

## Status

- [x] Data pipeline — `src/fetch_weo_data.py`
- [x] Forecasting module — `src/forecast.py` (ARIMA + VAR, with 95% CIs)
- [x] Anomaly detection — `src/anomaly_detection.py` (rolling-origin backtest)
- [x] IMF comparison — `src/compare_to_imf.py` (near-term hold-out check)
- [x] LLM briefing agent — `src/generate_briefing.py` (Claude API)
- [x] Visualization — `src/visualize.py`
- [x] Accuracy metrics — `src/model_accuracy.py` (MAE/RMSE)
- [x] Stationarity testing — `src/stationarity_test.py` (ADF)

## Charts

**Real GDP growth: actual history vs. ARIMA/VAR forecast, per country**
![Forecast fan chart](charts/forecast_fan.png)

**Where our model diverges from IMF's current outlook**
![IMF comparison heatmap](charts/imf_comparison_heatmap.png)

**Largest flagged deviations from trend (rolling-origin backtest)**
![Top anomalies](charts/top_anomalies.png)

## Project structure
## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/fetch_weo_data.py   # pulls WEO data -> data/weo_macro_indicators.csv
python src/forecast.py         # generates forecasts -> data/forecasts.csv
python src/anomaly_detection.py  # flags deviations -> data/anomalies.csv
python src/compare_to_imf.py     # near-term check -> data/imf_comparison.csv
export ANTHROPIC_API_KEY="sk-ant-..."  # get one at console.anthropic.com/settings/keys
python src/generate_briefing.py  # drafts a readable summary -> data/briefing.md
python src/visualize.py          # generates charts -> charts/*.png
python src/model_accuracy.py     # computes MAE/RMSE -> data/accuracy_report.csv
python src/stationarity_test.py  # ADF test -> data/stationarity_report.csv
```

`forecast.py` fits two models per country:
- **ARIMA** — one model per indicator, univariate, with the order (p,d,q)
  chosen per series by AIC (see `arima_utils.py`) instead of a fixed guess
- **VAR** — one model per country, jointly fitting all five indicators to
  capture cross-variable dynamics (e.g. how inflation and fiscal balance move
  together)

Both forecast types are written to `data/forecasts.csv` alongside the actuals,
tagged by `value_type`, along with `lower_95`/`upper_95` columns giving a 95%
confidence interval - "likely between X and Y," not just a single number.
The forecast fan chart shades these as bands so the growing uncertainty
further into the future is visible, not just implied.

`arima_utils.py` is a shared helper: instead of guessing ARIMA(1,1,1) for
every series, it tests a handful of candidate orders and picks whichever gets
the lowest AIC (a standard statistic that rewards fit while penalizing
needless complexity). Used by both `forecast.py` and `anomaly_detection.py`.
In `anomaly_detection.py`, the order is selected once per series from only
the initial training window - not the full series - so the choice can't
"see" the years it's later tested against.

`anomaly_detection.py` runs a rolling-origin backtest: at each year, it fits a
model on everything before it, forecasts one step ahead, and flags years where
the actual value deviates more than 2 standard deviations from what the model
expected. Note that WEO data blends true historical actuals with the IMF's own
forward projections with no flag distinguishing the two, so a flagged year
means "this diverged notably from a simple trend model" - worth a second look,
whether that's a real economic shock or a revised outlook.

`compare_to_imf.py` answers a different question: pretending we're standing at
the end of 2023, it forecasts 2024-2026 using only pre-2023 data, then compares
that forecast to what the IMF's own current WEO data shows for those same
years. This checks whether a simple model agrees with the IMF's own present-day
outlook, rather than testing against distant historical shocks.

`generate_briefing.py` is the agentic layer: it packages each country's latest
actuals, forecast trajectory, and any flagged anomalies into a prompt, sends it
to Claude, and writes the resulting plain-English briefing to `data/briefing.md`.
This turns raw model output into something a person (or a hiring manager
reading this repo) would actually want to read.

`model_accuracy.py` computes MAE (Mean Absolute Error) and RMSE (Root Mean
Squared Error) from both backtests: historical accuracy from
`anomaly_detection.py`'s rolling-origin test, and agreement with the IMF's
current outlook from `compare_to_imf.py`'s 2024-2026 hold-out. This replaces
"the model seems reasonable" with a specific, measured number - e.g. "our
1-year-ahead GDP growth forecasts are off by an average of X percentage
points historically." It also scores ARIMA against a naive baseline (predict
"no change from last year") per indicator - the standard sanity check for
whether the model is actually adding value over the simplest possible guess.

`stationarity_test.py` runs the Augmented Dickey-Fuller (ADF) test - the
standard first diagnostic step in time-series work - on every series, both at
the raw level and after one differencing pass. This is a formal statistical
cross-check on `arima_utils.py`'s AIC-based order selection: if a series is
non-stationary at the level but stationary once differenced, that confirms
d=1 was the statistically correct call, not just an AIC-preferred guess made
without a formal test behind it.

## Data source

[IMF DataMapper API](https://www.imf.org/external/datamapper/api/help) —
public, no API key required. Indicators and country coverage per the IMF
World Economic Outlook database.

## License

MIT
