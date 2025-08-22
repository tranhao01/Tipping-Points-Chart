# Singularity / Regime-Shift Analyzer (template + demo)
# ----------------------------------------------------
# What this cell does:
# 1) Creates a CSV template for your time series (time,value).
# 2) If you don't upload data, it generates a demo series.
# 3) Computes growth rate g(t), acceleration a(t).
# 4) Fits three models: Exponential, Logistic, Hyperbolic (finite-time).
# 5) Chooses best model by AIC/BIC; estimates t* if hyperbolic wins.
# 6) Runs a simple structural-break scan on ln(X) ~ t and early-warning stats.
# 7) Plots and writes a text report.
#
# Usage:
# - Download and fill '/mnt/data/time_series_template.csv' (two columns: time,value).
# - Re-run this cell after uploading your CSV to the same path OR change `data_path`.
#
# Notes:
# - No external internet or exotic libraries used.
# - Logistic fit uses a coarse grid search (K, r, c).
# - Hyperbolic fit uses a grid over t* and alpha with OLS on ln transforms.
# - Structural break uses a brute-force two-segment OLS comparison (Chow-like).

import os, math, io, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

TEMPLATE_PATH = "/mnt/data/time_series_template.csv"
REPORT_PATH   = "/mnt/data/singularity_report.txt"
FITS_PATH     = "/mnt/data/model_fits.csv"

# 1) Create a template CSV if not present
if not os.path.exists(TEMPLATE_PATH):
    template = pd.DataFrame({
        "time": list(range(0, 30)),
        "value": [np.nan]*30
    })
    template.to_csv(TEMPLATE_PATH, index=False)

# 2) Load data or create a demo series
data_path = TEMPLATE_PATH  # change if you uploaded somewhere else
df = pd.read_csv(data_path)

# If user hasn't filled data, generate a demo (logistic-ish) to illustrate
if df["value"].isna().all():
    np.random.seed(42)
    t = np.arange(0, 60)
    # demo logistic: X = K / (1 + c * exp(-r t)) + noise
    K, r, c = 100.0, 0.12, 15.0
    x = K / (1.0 + c * np.exp(-r * t))
    x = x + np.random.normal(scale=1.2, size=len(t))
    x = np.clip(x, 1e-6, None)
    df = pd.DataFrame({"time": t, "value": x})

# Ensure sorted and valid
df = df.dropna().copy()
df = df.sort_values("time")
df = df[df["value"] > 0]
df.reset_index(drop=True, inplace=True)

t = df["time"].values.astype(float)
x = df["value"].values.astype(float)
n = len(x)

# Helper: OLS y ~ [1, X]
def ols_fit(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    # add intercept
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_design = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    yhat = X_design @ beta
    resid = y - yhat
    rss = float(np.sum(resid**2))
    return beta, yhat, rss

def aic_bic(rss, n, k):
    # Gaussian MLE: AIC = 2k + n ln(RSS/n), BIC = k ln n + n ln(RSS/n)
    if n <= k or rss <= 0:
        return np.inf, np.inf
    aic = 2*k + n*np.log(rss/n)
    bic = k*np.log(n) + n*np.log(rss/n)
    return aic, bic

# 3) Growth g(t) and acceleration a(t)
logx = np.log(x)
g = np.diff(logx)  # discrete growth ~ d/dt ln X
a = np.diff(g)     # discrete acceleration

# 4) Model fits
fits_summary = []

# 4.1 Exponential: X = A * exp(b t)  => ln X = ln A + b t
beta_exp, yhat_exp_log, rss_exp_log = ols_fit(t, logx)
lnA, b = beta_exp
xhat_exp = np.exp(yhat_exp_log)
rss_exp = np.sum((x - xhat_exp)**2)
k_exp = 2  # (A, b)
aic_exp, bic_exp = aic_bic(rss_exp, n, k_exp)
fits_summary.append({"model":"exponential", "rss":rss_exp, "aic":aic_exp, "bic":bic_exp,
                     "params":{"A": float(np.exp(lnA)), "b": float(b)}})

# 4.2 Logistic: X = K / (1 + c * exp(-r t))
def logistic_curve(t, K, r, c):
    return K / (1.0 + c*np.exp(-r*t))

def fit_logistic_grid(t, x, K_grid, r_grid, c_grid):
    best = {"rss": np.inf, "K": None, "r": None, "c": None}
    for K in K_grid:
        for r in r_grid:
            for c in c_grid:
                xhat = logistic_curve(t, K, r, c)
                if np.any(~np.isfinite(xhat)) or np.any(xhat <= 0):
                    continue
                rss = float(np.sum((x - xhat)**2))
                if rss < best["rss"]:
                    best = {"rss": rss, "K": K, "r": r, "c": c}
    return best

# Define coarse grids based on data scale
xmax = float(np.max(x))
xmin = float(np.min(x))
K_grid = np.linspace(max(xmax, 1.1*xmax), 3.0*max(xmax, 1.0), 10)  # from ~max(x) to ~3*max(x)
r_grid = np.linspace(0.01, 1.0, 10)  # broad
# Estimate c from first point as starting magnitude, but we still grid
c_grid = np.logspace(-2, 2, 10)

best_log = fit_logistic_grid(t, x, K_grid, r_grid, c_grid)
k_log = 3  # K, r, c
aic_log, bic_log = aic_bic(best_log["rss"], n, k_log)
xhat_log = logistic_curve(t, best_log["K"], best_log["r"], best_log["c"]) if np.isfinite(best_log["rss"]) else np.full_like(x, np.nan)
fits_summary.append({"model":"logistic", "rss":best_log["rss"], "aic":aic_log, "bic":bic_log,
                     "params": {"K": best_log["K"], "r": best_log["r"], "c": best_log["c"]}})

# 4.3 Hyperbolic finite-time: X = C / (t* - t)^alpha
# => ln X = ln C - alpha * ln(t* - t). Grid over t*, alpha; OLS for lnC, alpha.
def fit_hyperbolic_grid(t, x, tstar_grid, alpha_grid):
    best = {"rss": np.inf, "tstar": None, "alpha": None, "lnC": None}
    for tstar in tstar_grid:
        dt = tstar - t
        if np.any(dt <= 0):  # t* must be strictly > all t
            continue
        lx = np.log(x)
        lx_dt = np.log(dt)
        # ln X = ln C - alpha * ln(t*-t) => y = a + b * z with a=lnC, b=(-alpha)
        beta, yhat, rss = ols_fit(lx_dt, lx)
        lnC, b = beta
        alpha = -b
        if alpha <= 0 or not np.isfinite(rss):
            continue
        if rss < best["rss"]:
            best = {"rss": float(rss), "tstar": float(tstar), "alpha": float(alpha), "lnC": float(lnC)}
    return best

tmax = np.max(t)
# let t* from (tmax + 1% span) up to (tmax + 2 * span)
span = max(5.0, tmax - np.min(t))
tstar_grid = np.linspace(tmax + 0.05*span, tmax + 2.0*span, 30)
alpha_grid = np.linspace(0.2, 3.0, 15)  # (not used directly; we infer alpha by OLS slope)
best_hyp = fit_hyperbolic_grid(t, x, tstar_grid, alpha_grid)
k_hyp = 3  # (C, alpha, t*)
aic_hyp, bic_hyp = aic_bic(best_hyp["rss"], n, k_hyp)

# Build fitted curve for hyperbolic
if best_hyp["tstar"] is not None:
    C = np.exp(best_hyp["lnC"])
    xhat_hyp = C / (best_hyp["tstar"] - t)**best_hyp["alpha"]
else:
    xhat_hyp = np.full_like(x, np.nan)

fits_summary.append({"model":"hyperbolic", "rss":best_hyp["rss"], "aic":aic_hyp, "bic":bic_hyp,
                     "params":{"C": float(np.exp(best_hyp["lnC"])) if best_hyp["lnC"] is not None else None,
                               "alpha": best_hyp["alpha"],
                               "tstar": best_hyp["tstar"]}})

fits_df = pd.DataFrame(fits_summary)
fits_df.to_csv(FITS_PATH, index=False)

# Choose best by BIC first (more conservative), then AIC
fits_df["rank_bic"] = fits_df["bic"].rank(method="dense")
fits_df["rank_aic"] = fits_df["aic"].rank(method="dense")
best_model_row = fits_df.sort_values(["bic","aic"]).iloc[0].to_dict()
best_model = best_model_row["model"]

# 5) Simple structural-break scan on ln(X) ~ t (two-segment OLS)
def two_segment_break_scan(t, y, min_seg=5):
    # returns best split index k (split between k and k+1), rss_single, rss_split, improvement
    _, _, rss_single = ols_fit(t, y)
    best = {"k": None, "rss_split": np.inf, "improve": -np.inf}
    for k in range(min_seg, len(y)-min_seg):
        t1, y1 = t[:k], y[:k]
        t2, y2 = t[k:], y[k:]
        _, _, rss1 = ols_fit(t1, y1)
        _, _, rss2 = ols_fit(t2, y2)
        rss_split = rss1 + rss2
        improve = rss_single - rss_split
        if improve > best["improve"]:
            best = {"k": k, "rss_split": float(rss_split), "improve": float(improve)}
    return best, float(rss_single)

break_result, rss_single = two_segment_break_scan(t, np.log(x), min_seg=max(5, n//10))
break_k = break_result["k"]
break_time = float(t[break_k]) if break_k is not None else None
break_improve = break_result["improve"] if break_k is not None else None

# 6) Early-warning stats on growth g(t)
def lag1_autocorr(series):
    s = np.asarray(series)
    if len(s) < 3:
        return np.nan
    s = s - np.mean(s)
    num = np.sum(s[1:]*s[:-1])
    den = np.sum(s[:-1]**2)
    return float(num/den) if den > 0 else np.nan

window = max(10, n//4)
ac_list, var_list, times_list = [], [], []
for i in range(window, len(g)+1):
    seg = g[i-window:i]
    ac_list.append(lag1_autocorr(seg))
    var_list.append(float(np.var(seg)))
    times_list.append(float(t[1:][i-1]))  # align to end of each window

# 7) Plots
plt.figure()
plt.plot(t, x, label="X(t)")
if best_model == "exponential":
    plt.plot(t, xhat_exp, label="exp fit")
elif best_model == "logistic":
    plt.plot(t, xhat_log, label="logistic fit")
elif best_model == "hyperbolic":
    plt.plot(t, xhat_hyp, label="hyperbolic fit")
plt.xlabel("time")
plt.ylabel("value")
plt.title("Time series and best-fit model")
plt.legend()
plt.show()

plt.figure()
plt.plot(t[1:], g)
plt.xlabel("time")
plt.ylabel("g(t) ≈ Δ ln X")
plt.title("Discrete growth rate g(t)")
plt.show()

plt.figure()
plt.plot(t[2:], a)
plt.xlabel("time")
plt.ylabel("a(t) ≈ Δg(t)")
plt.title("Discrete acceleration a(t)")
plt.show()

if len(ac_list) > 0:
    plt.figure()
    plt.plot(times_list, ac_list)
    plt.xlabel("time")
    plt.ylabel("rolling lag-1 autocorr of g(t)")
    plt.title(f"Early-warning: autocorr over window={window}")
    plt.show()

if len(var_list) > 0:
    plt.figure()
    plt.plot(times_list, var_list)
    plt.xlabel("time")
    plt.ylabel("rolling variance of g(t)")
    plt.title(f"Early-warning: variance over window={window}")
    plt.show()

# 8) Write report
def fmt_params(d):
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, (int,float)) and math.isfinite(v) else f"{k}={v}" for k,v in d.items())

report = io.StringIO()
report.write("=== Singularity / Regime-Shift Analyzer ===\n")
report.write(f"Generated at: {datetime.now()}\n\n")
report.write(f"Data points: n={n}\n")
report.write(f"Time range: [{t.min()} .. {t.max()}]\n\n")

report.write("Model comparison (lower is better):\n")
for row in fits_summary:
    report.write(f"- {row['model']}: RSS={row['rss']:.6g}, AIC={row['aic']:.6g}, BIC={row['bic']:.6g}, params: {fmt_params(row['params'])}\n")
report.write("\n")
report.write(f"Best model by BIC/AIC: {best_model}\n")

if best_model == "hyperbolic":
    report.write(f"Estimated t* (finite-time): {best_hyp['tstar']:.6g}\n")
    report.write(f"Alpha: {best_hyp['alpha']:.6g}; C: {math.exp(best_hyp['lnC']):.6g}\n")
    report.write("Note: Real systems often bend to logistic before reaching t*.\n")

report.write("\nStructural break (two-segment OLS on ln X ~ t):\n")
if break_k is not None:
    report.write(f"- Best split index k={break_k} at time≈{break_time}, improvement in RSS={break_improve:.6g}\n")
else:
    report.write("- No valid split found (data too short or uniform).\n")

if len(ac_list) > 0:
    report.write("\nEarly-warning (rolling on g(t)):\n")
    report.write(f"- Window size: {window}\n")
    report.write(f"- Last lag-1 autocorr: {ac_list[-1]:.4g}\n")
    report.write(f"- Last variance: {var_list[-1]:.4g}\n")
    report.write("Rising autocorr/variance may indicate approaching tipping.\n")

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report.getvalue())

# Display fits table to user
import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Model fit summary (downloadable CSV)", pd.read_csv(FITS_PATH))

print(f"Template CSV saved at: {TEMPLATE_PATH}")
print(f"Report saved at: {REPORT_PATH}")
print(f"Fits summary CSV at: {FITS_PATH}")
