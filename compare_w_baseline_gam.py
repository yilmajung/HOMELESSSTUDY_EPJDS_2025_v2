# baselines_four_models.py
# ------------------------------------------------------------
# Four baselines for city-level tent prediction:
#  1) Seasonal naive (lag-7)
#  2) Poisson GLM (L2)
#  3) LightGBM (Poisson objective) / XGBoost fallback
#  4) Spatiotemporal Poisson GAM (GAM-ST; pygam)
#  - Trains on labeled rows up to 2023-12-31, predicts ALL rows
#  - Aggregates to city level per day via Monte-Carlo Poisson (same as STVGP)
# ------------------------------------------------------------

# Cap threads BEFORE importing numpy/pandas (prevents libgomp errors) ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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

# Spatiotemporal GAM
try:
    from pygam import PoissonGAM, s, te
    HAS_PYGAM = True
except Exception:
    HAS_PYGAM = False

#############
# Config
#############
PARQUET = "data/main_daily_with_amenities.parquet"  # path to your enriched GeoParquet
DATE_COL = "date"
Y_COL    = "ground_truth"
ID_COL   = "bboxid"             # or "grid_id" if that's what you use

# Monte Carlo settings (match STVGP aggregation style)
S = 500
P_THRESH = [range(0.2, 0.8, 0.1)]
LAMBDA_THRESH = [-math.log(1.0 - p) for p in P_THRESH]

# Outputs
OUT_DAILY   = "city_daily_predictions_baselines_4models.csv"
OUT_SUMMARY = "baseline_city_metrics_4models.csv"

# Base covariates list
BASE_COVS = ["max","min","precipitation","total_population","white_ratio","black_ratio","hh_median_income"]

####################
# Helpers / metrics
####################
def sanitize_X(mat):
    X = np.asarray(mat, dtype=float)
    X[~np.isfinite(X)] = 0.0
    return X

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

#############################
# Load & feature engineering
#############################
print("Loading dataset…")
gdf = gpd.read_parquet(PARQUET)
df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

# Date
if DATE_COL not in df.columns:
    if "timestamp" in df.columns:
        df[DATE_COL] = pd.to_datetime(df["timestamp"], unit="s").dt.floor("D")
    else:
        raise ValueError("No 'date' or 'timestamp' column present.")
df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.floor("D")
df = df.sort_values([DATE_COL])

# Ensure lon/lat columns; derive from centroids if missing
if not {"latitude", "longitude"}.issubset(df.columns):
    cent = gdf.geometry.centroid
    df["latitude"]  = cent.y.values
    df["longitude"] = cent.x.values

# Base covariates presence
for c in BASE_COVS:
    if c not in df.columns: df[c] = 0.0

# Amenity features: log1p(n_*)
amen_cols = [c for c in df.columns if c.startswith("n_") and c != "n_amenities_total"]
for c in amen_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[f"log1p_{c}"] = np.log1p(df[c].astype(float))
AMEN_FEATS = [f"log1p_{c}" for c in amen_cols]

# Calendar features
df["dow"]     = df[DATE_COL].dt.weekday  # 0..6
df["month"]   = df[DATE_COL].dt.month    # 1..12
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

# Final design
X_cols = BASE_COVS + AMEN_FEATS + CAL_FEATS + LAG_FEATS

# Split for training learned models (GLM/GBM/GAM)
df_all = df.copy()  # keep all rows for prediction/aggregation
train_mask_time = df_all[DATE_COL] <= pd.Timestamp("2023-12-31")
train = df_all.loc[train_mask_time & df_all[Y_COL].notna()].copy()

# Design matrices
X_tr = sanitize_X(train[X_cols].values)
y_tr = train[Y_COL].values.astype(float)
X_all = sanitize_X(df_all[X_cols].values)

assert np.isfinite(y_tr).all(), "y_tr has NaNs or Infs"
assert (y_tr >= 0).all(), "y_tr has negatives (Poisson targets must be >= 0)"
assert np.isfinite(X_tr).all(), "X_tr has NaNs or Infs"

####################################
# Baseline 1: Seasonal naive (lag-7)
####################################
print("Predicting Seasonal Naive (lag-7)…")
df_all["lam_seasonal7"] = df_all["lag7"]
grid_mean = train.groupby(ID_COL)[Y_COL].mean()
nan_idx = df_all["lam_seasonal7"].isna() | ~np.isfinite(df_all["lam_seasonal7"])
if nan_idx.any():
    fill_vals = df_all.loc[nan_idx, ID_COL].map(grid_mean).fillna(train[Y_COL].mean())
    df_all.loc[nan_idx, "lam_seasonal7"] = fill_vals.values
df_all["lam_seasonal7"] = df_all["lam_seasonal7"].clip(lower=1e-6)

###############################
# Baseline 2: Poisson GLM (L2)
###############################
print("Training Poisson GLM (L2)…")
scaler = StandardScaler(with_mean=True, with_std=True)
X_tr_sc  = scaler.fit_transform(X_tr)
X_all_sc = scaler.transform(X_all)
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):  # extra safety for MKL/OpenBLAS threads
    poiss = PoissonRegressor(alpha=1.0, max_iter=300)
    poiss.fit(X_tr_sc, y_tr)
df_all["lam_poisson_glm"] = poiss.predict(X_all_sc).clip(min=1e-6)

# #######################################################
# Baseline 3: GBM (LightGBM preferred; XGBoost fallback)
# #######################################################
def fit_predict_gbm(Xtr, ytr, df_train_labeled, Xall):
    cutoff = df_train_labeled[DATE_COL].max() - pd.Timedelta(days=90)
    valid  = df_train_labeled.loc[df_train_labeled[DATE_COL] > cutoff].copy()
    Xval   = sanitize_X(valid[X_cols].values)
    yval   = valid[Y_COL].values.astype(float)

    if HAS_LGB:
        try:
            dtrain = lgb.Dataset(Xtr, label=ytr)
            dvalid = lgb.Dataset(Xval, label=yval, reference=dtrain)
            params = dict(
                objective="poisson", metric="poisson",
                learning_rate=0.05, num_leaves=63,
                min_data_in_leaf=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, num_threads=1,   # cap threads
            )
            callbacks = [lgb.early_stopping(200, first_metric_only=True),
                         lgb.log_evaluation(period=0)]
            gbm = lgb.train(params, dtrain, num_boost_round=5000,
                            valid_sets=[dvalid], valid_names=["valid"],
                            callbacks=callbacks)
            best_it = getattr(gbm, "best_iteration", None)
            lam_hat = gbm.predict(Xall, num_iteration=best_it)
            return lam_hat.clip(min=1e-6), "LightGBM-Poisson"
        except Exception as e:
            print(f"[LightGBM] fallback to XGBoost: {e}")

    if HAS_XGB:
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xval, label=yval)
        dall   = xgb.DMatrix(Xall)
        params = dict(
            objective="count:poisson", eval_metric="poisson-nloglik",
            eta=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            lambda_=1.0, tree_method="hist", nthread=1,   # cap threads
        )
        model = xgb.train(params, dtrain, num_boost_round=5000,
                          evals=[(dvalid, "valid")],
                          early_stopping_rounds=200, verbose_eval=False)
        lam_hat = model.predict(dall, iteration_range=(0, model.best_iteration+1))
        return lam_hat.clip(min=1e-6), "XGBoost-Poisson"

    return None, None

print("Training GBM (Poisson)…")
lam_gbm, gbm_name = fit_predict_gbm(X_tr, y_tr, train, X_all)
if gbm_name is not None:
    df_all["lam_gbm"] = lam_gbm
else:
    print("GBM not available — skipping.")

##################################################
# Baseline 4: Spatiotemporal Poisson GAM (GAM-ST)
##################################################
def fit_predict_gam_st(gdf, df, date_col="date", y_col="ground_truth"):
    if not HAS_PYGAM:
        raise RuntimeError("pygam not installed. Install with: pip install pygam")

    # Ensure lon/lat
    if not {"latitude","longitude"}.issubset(df.columns):
        cent = gdf.geometry.centroid
        df = df.copy()
        df["latitude"]  = cent.y.values
        df["longitude"] = cent.x.values

    # Time index (days since start)
    t0 = pd.to_datetime(df[date_col]).min()
    df["t_idx"] = (pd.to_datetime(df[date_col]) - t0).dt.days.astype(int)

    # Ensure weekly/monthly ints
    if "dow" not in df.columns:   df["dow"] = pd.to_datetime(df[date_col]).dt.weekday
    if "month" not in df.columns: df["month"] = pd.to_datetime(df[date_col]).dt.month

    # Build GAM matrix
    amen_feats = [c for c in df.columns if c.startswith("log1p_n_")]
    lag_feats  = [c for c in df.columns if c in ("lag7","lag28")]
    X_cols_gam = ["longitude","latitude","t_idx","dow","month"] + BASE_COVS + amen_feats + lag_feats
    for c in X_cols_gam:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

    train_mask = (pd.to_datetime(df[date_col]) <= pd.Timestamp("2023-12-31")) & df[y_col].notna()
    Xtr = df.loc[train_mask, X_cols_gam].values.astype(np.float32)
    ytr = df.loc[train_mask, y_col].values.astype(np.float32)
    Xall = df[X_cols_gam].values.astype(np.float32)

    # Terms: spatial smooth (lon,lat), time smooth, cyclic weekly/monthly
    terms = te(0,1, n_splines=30, spline_order=3) + \
            s(2, n_splines=20) + \
            s(3, n_splines=6,  periodic=True) + \
            s(4, n_splines=12, periodic=True)

    # Near-linear small-spline terms for remaining covariates
    for j in range(5, len(X_cols_gam)):
        terms += s(j, n_splines=2, spline_order=1)

    gam = PoissonGAM(terms, fit_intercept=True)
    lams = np.logspace(-2, 3, 6)   # quick smoothing search
    gam = gam.gridsearch(Xtr, ytr, lam=lams, progress=False)

    lam_hat = np.clip(gam.predict_mu(Xall), 1e-6, None)
    return lam_hat, "GAM_ST"

print("Training Spatiotemporal Poisson GAM (GAM-ST)…")
try:
    lam_gam, gam_name = fit_predict_gam_st(gdf, df_all, date_col=DATE_COL, y_col=Y_COL)
    df_all["lam_gam_st"] = lam_gam
except Exception as e:
    print(f"GAM-ST skipped: {e}")
    gam_name = None

#####################################################
# City-level MC aggregation for all models + metrics
#####################################################
city_truth = (df_all.loc[df_all[Y_COL].notna()]
                .groupby(DATE_COL, as_index=False)[Y_COL].sum()
                .rename(columns={Y_COL: "city_truth"}))

MODEL_LIST = [
    ("lam_seasonal7",   "SeasonalNaive_lag7"),
    ("lam_poisson_glm", "PoissonGLM_L2"),
]
if gbm_name is not None:
    MODEL_LIST.append(("lam_gbm", gbm_name))
if gam_name is not None:
    MODEL_LIST.append(("lam_gam_st", gam_name))

results = []
for col, name in MODEL_LIST:
    for lambda_ in LAMBDA_THRESH:
        print(f"Aggregating city-level predictions for {name} at lambda_thresh={lambda_:.4f}…")
        agg = aggregate_city_mc(df_all, col, S=S, lambda_thresh=lambda_)
        agg["model"] = name
        agg["lambda_thresh"] = lambda_
        agg = agg.merge(city_truth, on="date", how="left")
        results.append(agg)

city_daily = pd.concat(results, ignore_index=True).sort_values(["model","date"])

# Metrics only on dates with ground-truth
summary_rows = []
for name, sub in city_daily.groupby("model"):
    msk = sub["city_truth"].notna()
    y_true = sub.loc[msk, "city_truth"].values
    y_hat  = sub.loc[msk, "sim_mean"].values
    summary_rows.append({"model": name, **eval_all(y_true, y_hat)})

summary = pd.DataFrame(summary_rows).sort_values("NLPD").reset_index(drop=True)
print("\n=== City-level baseline summary (4 models) ===")
print(summary)

# Save outputs
city_daily.to_csv(OUT_DAILY, index=False)
summary.to_csv(OUT_SUMMARY, index=False)
print(f"\nWrote daily city predictions → {OUT_DAILY}")
print(f"Wrote metric summary        → {OUT_SUMMARY}")
