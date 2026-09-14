# Tourism, Hotel Occupancy & Airfare — VAR Analysis

What actually drives hotel occupancy and room pricing? A 4-variable Vector
Autoregression on daily Occupancy, Room Rate, Tourist Demand, and Airfare
indices (Jan 2018 – Dec 2023, n = 2,190), with Granger causality, impulse
response functions, forecast error variance decomposition, and an
out-of-sample backtest.

## Headline finding

Room Rate leads the system. It Granger-causes Occupancy, Demand and Airfare
(p < 0.001 on all three) and is Granger-caused by none of them — it explains
~96% of Occupancy's long-run forecast error variance. This is the reverse of
the standard assumption that demand drives price: in this data, rate-setting
moves first and occupancy follows.

**Caveat, stated up front:** this dataset is very likely synthetic (Section
2.1 of the report — a burn-in transient and an out-of-range occupancy value
give it away). The directional finding (rate leads occupancy) is real and
internally consistent; the specific magnitudes (96% FEVD, huge F-statistics)
are almost certainly larger than real hotel-market data would show, where
noise and omitted variables are bigger. Treat this as a methodology
demonstration, not a number for a business case.

## Two methodological decisions that matter more than the results

1. **Levels, not differences.** The class template (built for equity-index
   VARs) differences the data by default. This series doesn't need it — ADF
   strongly rejects a unit root and KPSS fails to reject stationarity, on
   all four series. Differencing an already-stationary, mean-reverting
   series would have been an over-differencing error. Full test results in
   Section 2 of the report.
2. **Burn-in removal.** The first ~30 observations are a simulation
   transient (occupancy starts at 65 and free-falls to roughly -1,000
   before settling into steady-state oscillation) — not a market event.
   Dropped before any estimation.

## Out-of-sample backtest

The forecast in the report is not just an in-sample fit. The last 10 days
were held out, the model was re-fit on the training data only (lag order
re-selected on train, not on the full sample), and the forecast was checked
against actuals and against a naive "tomorrow = today" benchmark.

| Series | VAR RMSE | Naive RMSE | VAR beats naive? |
|---|---|---|---|
| Occupancy | 16.38 | 27.88 | Yes — 41% lower error |
| Room Rate | 132.73 | 149.55 | Yes — 11% lower error |
| Tourist Demand | 23.35 | 11.90 | **No — 96% higher error** |
| Airfare | 19.18 | 31.50 | Yes — 39% lower error |

The model beats a trivial baseline on 3 of 4 series. It loses on Tourist
Demand in this holdout window — reported here rather than hidden, because a
model's failure mode is as informative as its success. See Section 11 of the
report for the full discussion.

## Repo structure

```
├── data/Tourism_Hotel.csv                    raw daily indices
├── notebooks/tourism_var_analysis.ipynb      exploratory notebook
├── src/VAR_pipeline.py                       end-to-end script (stationarity → VAR → Granger → IRF → FEVD → backtest)
├── report/Tourism_Hotel_VAR_Report.pdf       full write-up with management summary
├── outputs/                                  generated figures and tables
└── requirements.txt
```

## Reproduce it

```bash
pip install -r requirements.txt
python src/VAR_pipeline.py --data data/Tourism_Hotel.csv --out outputs/
```

## Limitations

- Likely synthetic data — see caveat above and Section 10 of the report.
- Granger causality is predictive precedence, not proof of structural
  causation. An omitted common driver (e.g. a pricing calendar) could
  produce the same pattern without Rate being the true structural cause.
- Only 4 variables modelled; a real hotel-revenue system also has
  seasonality, competitor pricing, and macro travel demand.
- Backtest uses a single 10-day holdout window. A rolling-origin backtest
  across multiple windows would be needed before trusting this for real
  decisions.
