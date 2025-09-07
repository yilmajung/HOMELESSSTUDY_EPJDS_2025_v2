# pip install pandas numpy geopandas scikit-learn lightgbm xgboost statsmodels pyarrow
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

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
PARQUET = "data/main_daily_with_amenities.parquet"  # adjust if needed
ID_COL   = None  # will auto-detect "bboxid" or "grid_id"
DATE_COL = "date"
Y_COL    = "ground_truth"

BASE_COVS = ["max","min","precipitation","total_population","white_ratio","black_ratio","hh_median_income"]

# Helpers: metrics
from scipy.special import gammaln

def mape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).clip(min=eps)
    # If you truly have no zeros you can remove eps, but this is safer.
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0

def mean_poisson_deviance(y, mu, eps=1e-12):
    # 2 * [ y*log(y/mu) - (y-mu) ] with y*log(y/mu)=0 when y=0
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float).clip(min=eps)
    term = np.zeros_like(y)
    nz = y > 0
    term[nz] = y[nz]*np.log(y[nz]/mu[nz])
    return np.mean(2.0*(term - (y - mu)))

def nlpd_poisson(y, lam, eps=1e-12):
    # Negative Log Predictive Density for Poisson
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float).clip(min=eps)
    # -log P(Y=y | lam) = lam - y*log lam + log(y!)
    return np.mean(lam - y*np.log(lam) + gammaln(y+1.0))

def eval_all(y_true, y_hat_mu):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_hat_mu)),
        "MAPE(%)": mape(y_true, y_hat_mu),
        "MeanPoissonDev": mean_poisson_deviance(y_true, y_hat_mu),
        "NLPD": nlpd_poisson(y_true, y_hat_mu),
    }

# Load & feature engineering
gdf = gpd.read_parquet(PARQUET)
if DATE_COL not in gdf.columns:
    if "timestamp" in gdf.columns:
        gdf[DATE_COL] = pd.to_datetime(gdf["timestamp"], unit="s").dt.floor("D")
    else:
        raise ValueError("No 'date' or 'timestamp' column present.")

gdf[DATE_COL] = pd.to_datetime(gdf[DATE_COL]).dt.floor("D")
gdf = gdf.sort_values([DATE_COL])

if Y_COL not in gdf.columns:
    raise ValueError(f"Missing '{Y_COL}' column.")

# ID (per-grid) column
if ID_COL is None:
    if "bboxid" in gdf.columns: ID_COL = "bboxid"
    else:
        raise ValueError("Couldn't find bbox id column; set ID_COL to your id field.")

# Keep only rows with target defined for training/testing evaluation
df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
df = df[~df[Y_COL].isna()].copy()

# Amenity features: all n_* (excluding total to avoid double-count)
amen_cols = [c for c in df.columns if c.startswith("n_") and c != "n_amenities_total"]
for c in amen_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[f"log1p_{c}"] = np.log1p(df[c].astype(float))

AMEN_FEATS = [f"log1p_{c}" for c in amen_cols]

# Calendar/cyclical features
df["dow"]  = df[DATE_COL].dt.weekday  # 0..6
df["month"] = df[DATE_COL].dt.month   # 1..12
# Cyclical encode (helps for linear models)
df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7.0)
df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7.0)
df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)

CAL_FEATS = ["dow_sin","dow_cos","month_sin","month_cos"]

# Lag features inside each grid (avoid leakage)
def add_lags(_df, id_col, y_col, lags=(7, 28)):
    _df = _df.sort_values([id_col, DATE_COL]).copy()
    for L in lags:
        _df[f"lag{L}"] = _df.groupby(id_col)[y_col].shift(L)
    return _df

df = add_lags(df, ID_COL, Y_COL, lags=(7, 28))
LAG_FEATS = ["lag7","lag28"]

# Replace remaining missing covariates with 0
for c in BASE_COVS:
    if c not in df.columns: df[c] = 0.0
for c in LAG_FEATS:
    df[c] = df[c].fillna(0.0)

# Final design
X_cols = BASE_COVS + AMEN_FEATS + CAL_FEATS + LAG_FEATS

# -------------------------
# Train / test split (time-based)
# Train: <= 2023-12-31, Test: 2024-01-01..2024-05-31
# -------------------------
train_mask = df[DATE_COL] <= pd.Timestamp("2023-12-31")
test_mask  = (df[DATE_COL] >= pd.Timestamp("2024-01-01")) & (df[DATE_COL] <= pd.Timestamp("2024-05-31"))

train = df.loc[train_mask].copy()
test  = df.loc[test_mask].copy()

X_tr = train[X_cols].values.astype(float)
y_tr = train[Y_COL].values.astype(float)
X_te = test[X_cols].values.astype(float)
y_te = test[Y_COL].values.astype(float)

# -------------------------
# Baseline 0: Seasonal naive (lag-7) within each grid
# -------------------------
# Prediction: for each (id, t in test), use y_{t-7}; fallback to per-grid train mean if missing
gkey = [ID_COL, DATE_COL]
lag7_col = "lag7"

# We already created lag7 for all rows; just take it for test
yhat_seasonal7 = test[lag7_col].values.copy()
# Fallbacks: per-grid mean in training
grid_mean = train.groupby(ID_COL)[Y_COL].mean()
nan_idx = np.isnan(yhat_seasonal7) | np.isinf(yhat_seasonal7)
if nan_idx.any():
    # map per-grid mean
    fill_vals = test.loc[nan_idx, ID_COL].map(grid_mean).fillna(train[Y_COL].mean())
    yhat_seasonal7[nan_idx] = fill_vals.values

metrics_seasonal7 = eval_all(y_te, yhat_seasonal7)
print("\nBaseline — Seasonal Naive (lag-7):")
for k,v in metrics_seasonal7.items(): print(f"  {k}: {v:.4f}")

# -------------------------
# Baseline 1: Poisson GLM (L2)
# -------------------------
scaler = StandardScaler(with_mean=True, with_std=True)
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

poiss = PoissonRegressor(alpha=1.0, max_iter=200, warm_start=False)
poiss.fit(X_tr_sc, y_tr)
lam_pred_poiss = poiss.predict(X_te_sc).clip(min=1e-6)  # predicted mean (lambda)

metrics_poiss = eval_all(y_te, lam_pred_poiss)
print("\nBaseline — Poisson GLM (L2):")
for k,v in metrics_poiss.items(): print(f"  {k}: {v:.4f}")

# -------------------------
# Baseline 2: GBM with Poisson objective
# -------------------------
def fit_predict_gbm(Xtr, ytr, Xte, yte):
    # Prefer LightGBM (faster, native Poisson); fallback to XGBoost Poisson
    if HAS_LGB:
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dvalid = lgb.Dataset(Xte, label=yte, reference=dtrain)
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
        lam_hat = gbm.predict(Xte, num_iteration=gbm.best_iteration)
        return lam_hat.clip(min=1e-6), "LightGBM-Poisson"
    elif HAS_XGB:
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xte, label=yte)
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
        lam_hat = model.predict(dvalid, iteration_range=(0, model.best_iteration+1))
        return lam_hat.clip(min=1e-6), "XGBoost-Poisson"
    else:
        raise RuntimeError("Neither LightGBM nor XGBoost is installed.")

try:
    lam_pred_gbm, gbm_name = fit_predict_gbm(X_tr, y_tr, X_te, y_te)
    metrics_gbm = eval_all(y_te, lam_pred_gbm)
    print(f"\nBaseline — {gbm_name}:")
    for k,v in metrics_gbm.items(): print(f"  {k}: {v:.4f}")
except RuntimeError as e:
    print(f"\nSkipped GBM baseline: {e}")

# -------------------------
# (Optional) Zero-Inflated NB (statsmodels) — skeleton
# -------------------------
# from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
# import statsmodels.api as sm
# exog = sm.add_constant(X_tr_sc)
# zinb = ZeroInflatedNegativeBinomialP(endog=y_tr, exog=exog, exog_infl=exog[:, :1], inflation='logit')
# res = zinb.fit(method='lbfgs', maxiter=200, disp=False)
# lam_zinb = res.predict(sm.add_constant(X_te_sc), which="mean").clip(min=1e-6)
# metrics_zinb = eval_all(y_te, lam_zinb)
# print("\nBaseline — Zero-Inflated NB:")
# for k,v in metrics_zinb.items(): print(f"  {k}: {v:.4f}")

# -------------------------
# Summary table
# -------------------------
rows = [("SeasonalNaive_lag7", metrics_seasonal7)]
rows.append(("PoissonGLM_L2", metrics_poiss))
if 'metrics_gbm' in locals():
    rows.append((gbm_name, metrics_gbm))
# if 'metrics_zinb' in locals():
#     rows.append(("ZINB", metrics_zinb))

summary = pd.DataFrame([{**{"model": name}, **m} for name, m in rows])
print("\n=== Baseline Summary ===")
print(summary.sort_values("NLPD").reset_index(drop=True))  # smaller NLPD is better

# Save summary to CSV for further analysis if needed
summary.to_csv("data/baseline_summary.csv", index=False)
