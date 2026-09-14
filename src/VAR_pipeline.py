"""
Tourism-Hotel VAR Pipeline
---------------------------
Levels are stationary (ADF strongly rejects unit root; KPSS fails to reject
stationarity) after dropping the simulation burn-in window. VAR is estimated
on LEVELS, not first differences -- differencing an already-stationary,
mean-reverting series would have been an over-differencing error.

At the end, the model is backtested out-of-sample: the last 10 days are held
out, the model is re-fit on the training data only, and the forecast is
checked against actuals and against a naive "tomorrow = today" benchmark.
"""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
import warnings, os
warnings.filterwarnings("ignore")

OUT = "outputs"               # relative folder -- created automatically
DATA = "Tourism_Hotel.csv"    # put the CSV in the same folder as this script
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.set_index("Date").sort_index().asfreq("D")

# Drop first 30 obs: simulation burn-in transient (65 -> steady state ~-1005)
BURNIN = 30
y = df.iloc[BURNIN:].copy()
short = {"Occupancy Index":"Occ", "Room Rate Index":"Rate",
         "Tourist Demand Index":"Demand", "Airfare Index":"Airfare"}
y = y.rename(columns=short)
cols = list(y.columns)

print("Post burn-in shape:", y.shape)

# ---------- stationarity on levels, post burn-in ----------
def adf_kpss(s, name):
    s = s.dropna()
    a_stat, a_p, *_ = adfuller(s, autolag="AIC")
    try:
        k_stat, k_p, *_ = kpss(s, regression="c", nlags="auto")
    except Exception:
        k_stat, k_p = np.nan, np.nan
    return {"series": name, "ADF_stat": a_stat, "ADF_p": a_p, "KPSS_stat": k_stat, "KPSS_p": k_p}

lvl = pd.DataFrame([adf_kpss(y[c], c) for c in cols])
lvl.to_csv(f"{OUT}/02_stationarity_levels_postburnin.csv", index=False)
print(lvl)

fig, axes = plt.subplots(len(cols), 1, figsize=(10, 9), sharex=True)
for ax, c in zip(axes, cols):
    ax.plot(y.index, y[c], color="#0746ab", lw=0.7)
    ax.set_title(c, fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/01_levels_postburnin.png", dpi=130); plt.close()

# ---------- lag selection ----------
model = VAR(y)
lag_sel = model.select_order(maxlags=10)
with open(f"{OUT}/05_lag_selection_levels.txt", "w") as f:
    f.write(str(lag_sel.summary()))
print(lag_sel.summary())
p = max(lag_sel.aic, 1)
print("Chosen lag (AIC):", p)

results = model.fit(p)
with open(f"{OUT}/06_var_summary_levels.txt", "w") as f:
    f.write(str(results.summary()))

# ---------- diagnostics ----------
diag = []
diag.append(f"Stable (statsmodels is_stable): {results.is_stable()}")
resid = results.resid
for c in resid.columns:
    lb = acorr_ljungbox(resid[c], lags=[10], return_df=True)
    diag.append(f"Ljung-Box(10) {c}: stat={lb['lb_stat'].values[0]:.2f} p={lb['lb_pvalue'].values[0]:.4f}")
for c in resid.columns:
    jb, jbp, sk, ku = jarque_bera(resid[c])
    diag.append(f"Jarque-Bera {c}: stat={jb:.2f} p={jbp:.4f} skew={sk:.3f} kurt={ku:.3f}")
with open(f"{OUT}/07_diagnostics_levels.txt", "w") as f:
    f.write("\n".join(diag))
print("\n".join(diag))

# ---------- Granger causality ----------
gc = []
for caused in cols:
    for causing in cols:
        if caused == causing: continue
        t = results.test_causality(caused, [causing], kind="f")
        gc.append((causing, caused, t.test_statistic, t.pvalue, t.pvalue < 0.05))
gc_df = pd.DataFrame(gc, columns=["causing", "caused", "F", "p", "significant_5pct"])
gc_df.to_csv(f"{OUT}/08_granger_causality_levels.csv", index=False)
print(gc_df.to_string())

# ---------- IRF ----------
irf = results.irf(24)
fig = irf.plot(orth=True, signif=0.05)
fig.set_size_inches(13, 13)
fig.savefig(f"{OUT}/09_irf_levels.png", dpi=130); plt.close(fig)

fig2 = irf.plot(response="Occ", orth=True, signif=0.05)
fig2.set_size_inches(10, 8)
fig2.savefig(f"{OUT}/10_irf_occ_response.png", dpi=130); plt.close(fig2)

fig3 = irf.plot(response="Rate", orth=True, signif=0.05)
fig3.set_size_inches(10, 8)
fig3.savefig(f"{OUT}/10b_irf_rate_response.png", dpi=130); plt.close(fig3)

# ---------- FEVD ----------
# NOTE: fevd.summary() prints directly and returns None -- str(fevd.summary())
# would write the literal text "None" to file. Capture the printed output instead.
import io, contextlib
fevd = results.fevd(24)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    fevd.summary()
with open(f"{OUT}/11_fevd_summary_levels.txt", "w") as f:
    f.write(buf.getvalue())
fig4 = fevd.plot(); fig4.set_size_inches(10, 10)
fig4.savefig(f"{OUT}/12_fevd_levels.png", dpi=130); plt.close(fig4)

occ_idx = cols.index("Occ")
fevd_occ = pd.DataFrame(fevd.decomp[occ_idx], columns=cols)
fevd_occ.index = range(1, len(fevd_occ)+1)
fevd_occ.index.name = "horizon"
fevd_occ.to_csv(f"{OUT}/13_fevd_occ_table.csv")
print(fevd_occ)

rate_idx = cols.index("Rate")
fevd_rate = pd.DataFrame(fevd.decomp[rate_idx], columns=cols)
fevd_rate.index = range(1, len(fevd_rate)+1)
fevd_rate.index.name = "horizon"
fevd_rate.to_csv(f"{OUT}/13b_fevd_rate_table.csv")
print(fevd_rate)

# ---------- Forecast (full-sample model) ----------
lag_order = results.k_ar
fc = results.forecast(y.values[-lag_order:], steps=10)
fc_df = pd.DataFrame(fc, columns=cols)
fc_df.to_csv(f"{OUT}/14_forecast_levels.csv", index=False)
print(fc_df)

# ================================================================
# BACKTEST -- does the forecast actually work on data it hasn't seen?
# Hold out the last 10 days, re-fit on the training data only,
# forecast forward, compare to a naive "tomorrow = today" benchmark.
# ================================================================
HOLDOUT = 10
train = y.iloc[:-HOLDOUT]
test = y.iloc[-HOLDOUT:]

bt_model = VAR(train)
bt_lag_sel = bt_model.select_order(maxlags=10)
bt_p = max(bt_lag_sel.aic, 1)
print("Backtest lag (chosen on TRAIN only):", bt_p)

bt_results = bt_model.fit(bt_p)
bt_lag_order = bt_results.k_ar
bt_fc = bt_results.forecast(train.values[-bt_lag_order:], steps=HOLDOUT)
bt_fc_df = pd.DataFrame(bt_fc, columns=cols, index=test.index)

naive = pd.DataFrame(np.tile(train.iloc[-1].values, (HOLDOUT, 1)), columns=cols, index=test.index)

def metrics(actual, pred, name):
    err = actual - pred
    rmse = np.sqrt((err**2).mean())
    mae = err.abs().mean()
    mape = (err.abs() / actual.abs()).mean() * 100
    return pd.Series({"RMSE": rmse, "MAE": mae, "MAPE_%": mape}, name=name)

rows = []
for c in cols:
    rows.append(metrics(test[c], bt_fc_df[c], f"{c} (VAR)"))
    rows.append(metrics(test[c], naive[c], f"{c} (naive)"))
backtest_report = pd.DataFrame(rows)
backtest_report.to_csv(f"{OUT}/15_backtest_metrics.csv")
print(backtest_report)

skill = pd.DataFrame({
    "VAR_RMSE": [metrics(test[c], bt_fc_df[c], "")["RMSE"] for c in cols],
    "Naive_RMSE": [metrics(test[c], naive[c], "")["RMSE"] for c in cols],
}, index=cols)
skill["VAR_beats_naive"] = skill["VAR_RMSE"] < skill["Naive_RMSE"]
skill["improvement_%"] = (1 - skill["VAR_RMSE"] / skill["Naive_RMSE"]) * 100
skill.to_csv(f"{OUT}/15b_backtest_skill_vs_naive.csv")
print(skill)

fig5, axes = plt.subplots(len(cols), 1, figsize=(10, 11), sharex=False)
hist = train.iloc[-40:]
for ax, c in zip(axes, cols):
    ax.plot(hist.index, hist[c], color="#333333", lw=1.0, label="History")
    ax.plot(test.index, test[c], color="#000000", lw=1.6, marker="o", ms=3, label="Actual (holdout)")
    ax.plot(test.index, bt_fc_df[c], color="#0746ab", lw=1.6, marker="o", ms=3, label="VAR forecast")
    ax.plot(test.index, naive[c], color="#c0392b", lw=1.2, ls="--", label="Naive benchmark")
    ax.set_title(c, fontsize=10, loc="left")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/16_backtest_plot.png", dpi=130)
plt.close()

print("DONE")
