# pip install pandas numpy geopandas scikit-learn lightgbm xgboost pyarrow
import math
import os
# Cap all math/thread pools to avoid libgomp failures on HPCs
os.environ["OMP_NUM_THREADS"] = "1"          # OpenMP (libgomp / llvm-openmp)
os.environ["OPENBLAS_NUM_THREADS"] = "1"     # NumPy/OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"          # Intel MKL (if used)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # Apple Accelerate (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # numexpr, if present
# Optional: silence tokenizers in some envs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error

# Optional GBMs
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Config
PARQUET = "data/main_daily_with_amenities.parquet"
DATE_COL = "date"
Y_COL    = "ground_truth"
ID_COL   = "bboxid"
OUT_DAILY = "city_daily_predictions_baselines.csv"
OUT_SUMMARY = "baseline_city_metrics.csv"

# Monte Carlo settings (match the STVGP aggregation style)
S = 500
P_THRESH = 0.7
LAMBDA_THRESH = -math.log(1.0 - P_THRESH)  # P(Y>0)=1-exp(-lambda) >= p_thresh  <=> lambda >= -log(1-p)

# Metrics
def mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=eps)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def mean_poisson_deviance(y, mu, eps=1e-12):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float).clip(min=eps)
    term = np.zeros_like(y)
    nz = y > 0
    term[nz] = y[nz]*np.log(y[nz]/mu[nz])
    return np.mean(2.0*(term - (y - mu)))

def nlpd_poisson(y, lam, eps=1e-12):
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float).clip(min=eps)
    return np.mean(lam - y*np.log(lam) + gammaln(y+1.0))

def eval_all(y_true, y_hat_mu):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_hat_mu)),
        "MAPE(%)": mape(y_true, y_hat_mu),
        "MeanPoissonDev": mean_poisson_deviance(y_true, y_hat_mu),
        "NLPD": nlpd_poisson(y_true, y_hat_mu),
    }

# Load & features
gdf = gpd.read_parquet(PARQUET)
df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

if DATE_COL not in df.columns:
    if "timestamp" in df.columns:
        df[DATE_COL] = pd.to_datetime(df["timestamp"], unit="s").dt.floor("D")
    else:
        raise ValueError("No 'date' or 'timestamp' column present.")
df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.floor("D")
df = df.sort_values([DATE_COL])

if Y_COL not in df.columns:
    raise ValueError(f"Missing '{Y_COL}' column.")

# Base covariates (ensure presence)
BASE_COVS = ["max","min","precipitation","total_population","white_ratio","black_ratio","hh_median_income"]
for c in BASE_COVS:
    if c not in df.columns: df[c] = 0.0

# Amenity features: log1p(n_*)
amen_cols = [c for c in df.columns if c.startswith("n_") and c != "n_amenities_total"]
for c in amen_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[f"log1p_{c}"] = np.log1p(df[c].astype(float))
AMEN_FEATS = [f"log1p_{c}" for c in amen_cols]

# Calendar features
df["dow"]  = df[DATE_COL].dt.weekday  # 0..6
df["month"] = df[DATE_COL].dt.month   # 1..12
df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7.0)
df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7.0)
df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
CAL_FEATS = ["dow_sin","dow_cos","month_sin","month_cos"]

# Lag features within each grid
def add_lags(_df, id_col, y_col, lags=(7, 28)):
    _df = _df.sort_values([id_col, DATE_COL]).copy()
    for L in lags:
        _df[f"lag{L}"] = _df.groupby(id_col)[y_col].shift(L)
    return _df

df = add_lags(df, ID_COL, Y_COL, lags=(7, 28))
LAG_FEATS = ["lag7","lag28"]
for c in LAG_FEATS:
    df[c] = df[c].fillna(0.0)

X_cols = BASE_COVS + AMEN_FEATS + CAL_FEATS + LAG_FEATS

# -------------------------
# Train / test split (time-based) with strict label filtering
# -------------------------
df_all = df.copy()  # keep everything (labels may be NaN)
train_mask_time = df_all[DATE_COL] <= pd.Timestamp("2023-12-31")
test_mask_time  = (df_all[DATE_COL] >= pd.Timestamp("2024-01-01")) & (df_all[DATE_COL] <= pd.Timestamp("2024-05-31"))

# Only rows with labels for fitting/evaluating learned models
train = df_all.loc[train_mask_time & df_all[Y_COL].notna()].copy()
test  = df_all.loc[test_mask_time  & df_all[Y_COL].notna()].copy()

def sanitize_X(mat):
    X = np.asarray(mat, dtype=float)
    X[~np.isfinite(X)] = 0.0
    return X

# Design matrices
X_tr = sanitize_X(train[X_cols].values)
y_tr = train[Y_COL].values.astype(float)
X_te = sanitize_X(test[X_cols].values)
y_te = test[Y_COL].values.astype(float)

# Extra checks (will raise if any issue remains)
assert np.isfinite(y_tr).all(), "y_tr has NaNs or Infs"
assert (y_tr >= 0).all(), "y_tr has negatives (Poisson targets must be >= 0)"
assert np.isfinite(X_tr).all(), "X_tr has NaNs or Infs"

# For predicting the entire dataset later (including unlabeled rows)
X_all = sanitize_X(df_all[X_cols].values)

# -------------------------
# Baseline 0: Seasonal naive (lag-7) — works on full df_all
# -------------------------
print("Predicting Seasonal Naive (lag-7)…")
df_all["lam_seasonal7"] = df_all["lag7"]
grid_mean = train.groupby(ID_COL)[Y_COL].mean()

nan_idx = df_all["lam_seasonal7"].isna() | ~np.isfinite(df_all["lam_seasonal7"])
if nan_idx.any():
    fill_vals = df_all.loc[nan_idx, ID_COL].map(grid_mean).fillna(train[Y_COL].mean())
    df_all.loc[nan_idx, "lam_seasonal7"] = fill_vals.values
df_all["lam_seasonal7"] = df_all["lam_seasonal7"].clip(lower=1e-6)

# -------------------------
# Baseline 1: Poisson GLM (L2)
# -------------------------
print("Training Poisson GLM (L2)…")
scaler = StandardScaler(with_mean=True, with_std=True)
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)
X_all_sc = scaler.transform(X_all)

poiss = PoissonRegressor(alpha=1.0, max_iter=300)
poiss.fit(X_tr_sc, y_tr)
df_all["lam_poisson_glm"] = poiss.predict(X_all_sc).clip(min=1e-6)

# -------------------------
# Baseline 2: GBM with Poisson objective (LightGBM/XGBoost)
# -------------------------
# fallback: sklearn API
from lightgbm import LGBMRegressor

def fit_predict_gbm(Xtr, ytr, df_train_labeled, Xall, date_col=DATE_COL, y_col=Y_COL):
    # validation = last 90 days of the labeled training window (no leakage)
    cutoff = df_train_labeled[date_col].max() - pd.Timedelta(days=90)
    valid = df_train_labeled.loc[df_train_labeled[date_col] > cutoff].copy()
    Xval = sanitize_X(valid[X_cols].values)
    yval = valid[y_col].values.astype(float)

    if HAS_LGB:
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dvalid = lgb.Dataset(Xval, label=yval, reference=dtrain)

        params = dict(
            objective="poisson",
            metric="poisson",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            # no 'verbose_eval' here; we'll control logs via callbacks
        )

        # Use callbacks for early stopping (works across LGB versions)
        callbacks = [
            lgb.early_stopping(stopping_rounds=200, first_metric_only=True),
            lgb.log_evaluation(period=0),  # silent; change to e.g. 50 for logs
        ]

        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )
        best_it = getattr(gbm, "best_iteration", None)
        lam_hat = gbm.predict(Xall, num_iteration=best_it)
        return lam_hat.clip(min=1e-6), "LightGBM-Poisson"

    elif HAS_XGB:
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        cutoff = df_train_labeled[date_col].max() - pd.Timedelta(days=90)
        valid = df_train_labeled.loc[df_train_labeled[date_col] > cutoff].copy()
        Xval = sanitize_X(valid[X_cols].values)
        yval = valid[y_col].values.astype(float)
        dvalid = xgb.DMatrix(Xval, label=yval)
        dall   = xgb.DMatrix(Xall)

        params = dict(
            objective="count:poisson",
            eval_metric="poisson-nloglik",
            eta=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            lambda_=1.0,
            tree_method="hist",
        )
        evallist = [(dvalid, "valid")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=evallist,
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        lam_hat = model.predict(dall, iteration_range=(0, model.best_iteration+1))
        return lam_hat.clip(min=1e-6), "XGBoost-Poisson"

    else:
        return None, None


print("Training GBM (Poisson)…")
lam_gbm, gbm_name = fit_predict_gbm(X_tr, y_tr, train, X_all)
if gbm_name is not None:
    df_all["lam_gbm"] = lam_gbm
else:
    print("GBM not available — skipping.")

# -------------------------
# City-level MC aggregation (unchanged), but use df_all now
# -------------------------
def aggregate_city_mc(df_in, lam_col, S=500, lambda_thresh=LAMBDA_THRESH):
    la = df_in[lam_col].values.astype(float)
    mask = la >= lambda_thresh
    df_use = df_in.loc[mask, [DATE_COL, lam_col]].copy()

    out_rows = []
    for date, sub in df_use.groupby(DATE_COL, sort=True):
        lam = sub[lam_col].values.astype(float)
        exp_total = lam.sum()
        sims = np.random.poisson(lam, size=(S, lam.size)).sum(axis=1)
        out_rows.append((date, exp_total, sims.mean(), *np.quantile(sims, [0.05, 0.50, 0.95]), lam.size))
    out = pd.DataFrame(out_rows, columns=["date","expected_total","sim_mean","sim_q05","sim_q50","sim_q95","active_boxes"])
    return out.sort_values("date").reset_index(drop=True)

# Ground truth for metric calc (sum over boxes) — only where labels exist
city_truth = df_all.loc[df_all[Y_COL].notna()].groupby(DATE_COL, as_index=False)[Y_COL].sum() \
                  .rename(columns={Y_COL: "city_truth"})


# ============================================================
# Threshold sweep ONLY for GLM and Naive (not GBM)
# ============================================================
P_THRESH_LIST = [0.2, 0.3, 0.4, 0.5, 0.6]   # values to test

def sweep_thresholds(df_all, models, p_list, S=500):
    """
    For each (model, p_thresh), aggregate city totals using MC with
    lambda_thresh = -log(1 - p_thresh). Returns (daily_df, summary_df).
    """
    all_daily = []
    all_summary = []

    # Ensure we have ground-truth city totals for metrics
    ct = df_all.loc[df_all[Y_COL].notna()] \
               .groupby(DATE_COL, as_index=False)[Y_COL].sum() \
               .rename(columns={Y_COL: "city_truth"})

    for p in p_list:
        lam_thresh = -math.log(1.0 - p)
        for lam_col, model_name in models:
            # City aggregation via MC (keeps parity with your STVGP eval)
            agg = aggregate_city_mc(df_all, lam_col, S=S, lambda_thresh=lam_thresh)
            agg["model"] = model_name
            agg["p_thresh"] = p
            agg = agg.merge(ct, on="date", how="left")

            # Metrics (compare sim_mean vs truth on days with labels)
            msk = agg["city_truth"].notna()
            y_true = agg.loc[msk, "city_truth"].values
            y_hat  = agg.loc[msk, "sim_mean"].values
            m = eval_all(y_true, y_hat)
            all_summary.append({"model": model_name, "p_thresh": p, **m})
            all_daily.append(agg)

    daily_df  = pd.concat(all_daily,  ignore_index=True) \
                  .sort_values(["model", "p_thresh", "date"])
    summary_df = pd.DataFrame(all_summary) \
                   .sort_values(["model", "p_thresh"]) \
                   .reset_index(drop=True)
    return daily_df, summary_df

# Which models to evaluate in the sweep
MODELS_TO_SWEEP = [
    ("lam_seasonal7",   "SeasonalNaive_lag7"),
    ("lam_poisson_glm", "PoissonGLM_L2"),
]

print("\nRunning P_THRESH sweep for Naive & GLM…")
daily_sweep, summary_sweep = sweep_thresholds(
    df_all=df_all,
    models=MODELS_TO_SWEEP,
    p_list=P_THRESH_LIST,
    S=S,                           # reuse your MC sample size
)

# Save sweep outputs
SWEEP_DAILY_OUT   = "city_daily_predictions_threshold_sweep.csv"
SWEEP_SUMMARY_OUT = "baseline_city_metrics_threshold_sweep.csv"

daily_sweep.to_csv(SWEEP_DAILY_OUT, index=False)
summary_sweep.to_csv(SWEEP_SUMMARY_OUT, index=False)

print("\n=== Threshold sweep summary (Naive & GLM) ===")
print(summary_sweep)
print(f"Wrote threshold-sweep daily predictions → {SWEEP_DAILY_OUT}")
print(f"Wrote threshold-sweep metric summary  → {SWEEP_SUMMARY_OUT}")
