# pip install pandas numpy geopandas scikit-learn lightgbm xgboost pyarrow
import os, math
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

# Train / test split for fitting models (no leakage)
train_mask = df[DATE_COL] <= pd.Timestamp("2023-12-31")
test_mask  = (df[DATE_COL] >= pd.Timestamp("2024-01-01")) & (df[DATE_COL] <= pd.Timestamp("2024-05-31"))
train = df.loc[train_mask].copy()
test  = df.loc[test_mask].copy()

# Arrays
X_tr = train[X_cols].values.astype(float)
y_tr = train[Y_COL].values.astype(float)
X_all = df[X_cols].values.astype(float)
y_all = df[Y_COL].values.astype(float)

# -------------------------
# Baseline 0: Seasonal naive (lag-7)
# -------------------------
print("Predicting Seasonal Naive (lag-7)…")
# Use the already-computed lag7 as prediction; for missing, fallback to per-grid train mean
df["lam_seasonal7"] = df["lag7"].values
grid_mean = train.groupby(ID_COL)[Y_COL].mean()
nan_idx = df["lam_seasonal7"].isna() | np.isinf(df["lam_seasonal7"])
if nan_idx.any():
    fill_vals = df.loc[nan_idx, ID_COL].map(grid_mean).fillna(train[Y_COL].mean())
    df.loc[nan_idx, "lam_seasonal7"] = fill_vals.values
df["lam_seasonal7"] = df["lam_seasonal7"].clip(lower=1e-6)

# -------------------------
# Baseline 1: Poisson GLM (L2)
# -------------------------
print("Training Poisson GLM (L2)…")
scaler = StandardScaler(with_mean=True, with_std=True)
X_tr_sc = scaler.fit_transform(X_tr)
X_all_sc = scaler.transform(X_all)
poiss = PoissonRegressor(alpha=1.0, max_iter=300)
poiss.fit(X_tr_sc, y_tr)
df["lam_poisson_glm"] = poiss.predict(X_all_sc).clip(min=1e-6)

# -------------------------
# Baseline 2: GBM with Poisson objective (LightGBM or XGBoost)
# -------------------------
def fit_predict_gbm(Xtr, ytr, Xall):
    # Use the tail of train as validation (last 90 days) for early stopping
    # Build a boolean mask for train val split
    tr_idx = train.index.values
    # last 90 days within train:
    cutoff = train[DATE_COL].max() - pd.Timedelta(days=90)
    valid_mask = train[DATE_COL] > cutoff
    Xval = train.loc[valid_mask, X_cols].values.astype(float)
    yval = train.loc[valid_mask, Y_COL].values.astype(float)

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
            verbose=-1,
        )
        gbm = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        lam_hat = gbm.predict(Xall, num_iteration=gbm.best_iteration)
        return lam_hat.clip(min=1e-6), "LightGBM-Poisson"
    elif HAS_XGB:
        dtrain = xgb.DMatrix(Xtr, label=ytr)
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
        evallist = [(dvalid, 'valid')]
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
lam_gbm, gbm_name = fit_predict_gbm(X_tr, y_tr, X_all)
if gbm_name is not None:
    df["lam_gbm"] = lam_gbm
else:
    print("GBM not available — skipping.")


# City-level Monte Carlo aggregation (per model)
print("Aggregating city-level predictions via Monte Carlo simulation…")
def aggregate_city_mc(df_in, lam_col, S=500, lambda_thresh=LAMBDA_THRESH):
    # Filter by P(Y>0) >= p_thresh  <=> lambda >= lambda_thresh
    la = df_in[lam_col].values.astype(float)
    mask = la >= lambda_thresh
    df_use = df_in.loc[mask, [DATE_COL, lam_col]].copy()

    out_rows = []
    for date, sub in df_use.groupby(DATE_COL, sort=True):
        lam = sub[lam_col].values.astype(float)
        if lam.size == 0:
            out_rows.append((date, 0.0, 0.0, 0.0, 0.0, 0, 0.0))
            continue
        exp_total = lam.sum()
        # MC simulate city total ~ sum_i Poisson(lam_i)
        # vectorized: shape (S, n)
        sims = np.random.poisson(lam, size=(S, lam.size)).sum(axis=1)
        mean_sim = sims.mean()
        q05, q50, q95 = np.quantile(sims, [0.05, 0.50, 0.95])
        out_rows.append((date, exp_total, mean_sim, q05, q95, lam.size, q50))
    out = pd.DataFrame(out_rows, columns=["date","expected_total","sim_mean","sim_q05","sim_q95","active_boxes","sim_q50"])
    out = out.sort_values("date").reset_index(drop=True)
    return out

city_truth = df.groupby(DATE_COL, as_index=False)[Y_COL].sum().rename(columns={Y_COL: "city_truth"})
results = []

for model_col, model_name in [
    ("lam_seasonal7","SeasonalNaive_lag7"),
    ("lam_poisson_glm","PoissonGLM_L2"),
    ("lam_gbm", gbm_name if gbm_name is not None else None)
]:
    if model_name is None:
        continue
    print(f"Aggregating city totals via MC: {model_name} …")
    agg = aggregate_city_mc(df, model_col, S=S, lambda_thresh=LAMBDA_THRESH)
    agg["model"] = model_name
    # Merge ground-truth (use only dates where truth exists)
    agg = agg.merge(city_truth, on="date", how="left")
    results.append(agg)

city_daily = pd.concat(results, ignore_index=True).sort_values(["model","date"])

# Metrics at city-level (compare sim_mean to truth over available days)
summary_rows = []
for model_name, sub in city_daily.groupby("model"):
    eval_mask = ~sub["city_truth"].isna()
    y_true = sub.loc[eval_mask, "city_truth"].values
    y_hat  = sub.loc[eval_mask, "sim_mean"].values  # could also compare 'expected_total'
    m = eval_all(y_true, y_hat)
    summary_rows.append({"model": model_name, **m})

summary = pd.DataFrame(summary_rows).sort_values("NLPD").reset_index(drop=True)

# Save outputs
city_daily.to_csv(OUT_DAILY, index=False)
summary.to_csv(OUT_SUMMARY, index=False)

print("\n=== City-level baseline summary (lower is better for all metrics) ===")
print(summary)
print(f"\nWrote daily city predictions to: {OUT_DAILY}")
print(f"Wrote metric summary to: {OUT_SUMMARY}")
