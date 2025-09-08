# compare_w_baseline.py

import os
import math
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.special import gammaln

# CONFIG
PARQUET = "data/main_daily_with_amenities.parquet"
DATE_COL = "date"
Y_COL    = "ground_truth"
ID_COL   = "bboxid"

# Features
BASE_COVS = ["max","min","precipitation","total_population","white_ratio","black_ratio","hh_median_income"]
LAGS      = (7, 28)     # per-grid lags
USE_AMENITIES = True    # use log1p(n_*) features

# Aggregation / MC
P_THRESH = 0.69
LAMBDA_THRESH = -np.log(1.0 - P_THRESH)  # keep cells with P(Y>0)>=P_THRESH
USE_EXACT_AGG = True     # exact Poisson sum
MC_SAMPLES     = 500     # used only if USE_EXACT_AGG=False
MC_USE_TORCH   = True    # GPU MC with PyTorch if available (else numpy)

# Output
OUT_PER_CELL = "per_cell_lambdas.csv"
OUT_CITY     = "city_daily_predictions_gbm.csv"
OUT_SUMMARY  = "baseline_city_metrics_gbm.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# UTILITIES
def sanitize_X(mat: np.ndarray) -> np.ndarray:
    X = np.asarray(mat, dtype=np.float32)
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
        "RMSE": float(np.sqrt(np.mean((y_true - y_hat_mu)**2))),
        "MAPE(%)": float(mape(y_true, y_hat_mu)),
        "MeanPoissonDev": float(mean_poisson_deviance(y_true, y_hat_mu)),
        "NLPD": float(nlpd_poisson(y_true, y_hat_mu)),
    }

def add_lags(df: pd.DataFrame, id_col: str, y_col: str, lags=(7,)) -> pd.DataFrame:
    df = df.sort_values([id_col, DATE_COL]).copy()
    for L in lags:
        df[f"lag{L}"] = df.groupby(id_col)[y_col].shift(L)
    return df

def build_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = df_raw.copy()
    # date
    if DATE_COL not in df.columns:
        if "timestamp" in df.columns:
            df[DATE_COL] = pd.to_datetime(df["timestamp"], unit="s").dt.floor("D")
        else:
            raise ValueError("No 'date' or 'timestamp' column present.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.floor("D")

    # ensure base covariates exist
    for c in BASE_COVS:
        if c not in df.columns: df[c] = 0.0

    # amenities → log1p(n_*)
    amen_feats = []
    if USE_AMENITIES:
        amen_cols = [c for c in df.columns if c.startswith("n_") and c != "n_amenities_total"]
        for c in amen_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            newc = f"log1p_{c}"
            df[newc] = np.log1p(df[c].astype(float))
            amen_feats.append(newc)

    # calendar
    df["dow"]   = df[DATE_COL].dt.weekday
    df["month"] = df[DATE_COL].dt.month
    df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7.0)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
    cal_feats = ["dow_sin","dow_cos","month_sin","month_cos"]

    # lags per grid
    df = add_lags(df, ID_COL, Y_COL, LAGS)
    for L in LAGS:
        df[f"lag{L}"] = df[f"lag{L}"].fillna(0.0)
    lag_feats = [f"lag{L}" for L in LAGS]

    X_cols = BASE_COVS + amen_feats + cal_feats + lag_feats
    # finalize dtype
    for c in X_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32).fillna(0)

    return df, X_cols

# MODEL: LightGBM/XGBoost (GPU-first)
def fit_predict_gbm(Xtr, ytr, Xall, df_train_labeled: pd.DataFrame, X_cols: list) -> tuple[np.ndarray, str]:
    """
    Prefers LightGBM on GPU, then XGBoost on GPU; cleanly falls back to CPU.
    Returns (lambda_pred_all, model_name).
    """
    cutoff = df_train_labeled[DATE_COL].max() - pd.Timedelta(days=90)
    valid = df_train_labeled.loc[df_train_labeled[DATE_COL] > cutoff].copy()
    Xval = sanitize_X(valid[X_cols].values)
    yval = valid[Y_COL].values.astype(np.float32)

    # Try LightGBM (GPU → CPU)
    try:
        import lightgbm as lgb
        params = dict(
            objective="poisson",
            metric="poisson",
            learning_rate=0.07,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
        )
        # GPU hint
        params.update({"device_type": "gpu", "max_bin": 255})

        dtrain = lgb.Dataset(Xtr, label=ytr)
        dvalid = lgb.Dataset(Xval, label=yval, reference=dtrain)
        callbacks = [lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(period=0)]
        t0 = time.time()
        gbm = lgb.train(params, dtrain, num_boost_round=5000, valid_sets=[dvalid], valid_names=["valid"], callbacks=callbacks)
        t1 = time.time()
        best_it = getattr(gbm, "best_iteration", None)
        lam_hat = gbm.predict(Xall, num_iteration=best_it).clip(min=1e-6)
        print(f"[LightGBM-GPU] trained in {t1 - t0:.1f}s, best_iter={best_it}")
        return lam_hat.astype(np.float64), "LightGBM-Poisson-GPU"
    except Exception as e:
        print(f"[LightGBM-GPU] fallback: {e}")

    # Try XGBoost (GPU → CPU)
    try:
        import xgboost as xgb
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        # Use last-90 days as validation
        dvalid = xgb.DMatrix(Xval, label=yval)
        dall   = xgb.DMatrix(Xall)
        params = dict(
            objective="count:poisson",
            eval_metric="poisson-nloglik",
            eta=0.07,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            lambda_=1.0,
            tree_method="gpu_hist",       # GPU first
            predictor="gpu_predictor",
            max_bin=256,
        )
        t0 = time.time()
        model = xgb.train(params, dtrain, num_boost_round=5000, evals=[(dvalid, "valid")],
                          early_stopping_rounds=200, verbose_eval=False)
        t1 = time.time()
        lam_hat = model.predict(dall, iteration_range=(0, model.best_iteration+1)).clip(min=1e-6)
        print(f"[XGBoost-GPU] trained in {t1 - t0:.1f}s, best_iter={model.best_iteration}")
        return lam_hat.astype(np.float64), "XGBoost-Poisson-GPU"
    except Exception as e:
        print(f"[XGBoost-GPU] fallback: {e}")

    # LightGBM CPU
    try:
        import lightgbm as lgb
        params = dict(
            objective="poisson",
            metric="poisson",
            learning_rate=0.07,
            num_leaves=63,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            device_type="cpu",
        )
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dvalid = lgb.Dataset(Xval, label=yval, reference=dtrain)
        callbacks = [lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(period=50)]
        t0 = time.time()
        gbm = lgb.train(params, dtrain, num_boost_round=5000, valid_sets=[dvalid], valid_names=["valid"], callbacks=callbacks)
        t1 = time.time()
        best_it = getattr(gbm, "best_iteration", None)
        lam_hat = gbm.predict(Xall, num_iteration=best_it).clip(min=1e-6)
        print(f"[LightGBM-CPU] trained in {t1 - t0:.1f}s, best_iter={best_it}")
        return lam_hat.astype(np.float64), "LightGBM-Poisson-CPU"
    except Exception as e:
        print(f"[LightGBM-CPU] fallback: {e}")

    # XGBoost CPU
    try:
        import xgboost as xgb
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dvalid = xgb.DMatrix(Xval, label=yval)
        dall   = xgb.DMatrix(Xall)
        params = dict(
            objective="count:poisson",
            eval_metric="poisson-nloglik",
            eta=0.07,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            lambda_=1.0,
            tree_method="hist",
        )
        t0 = time.time()
        model = xgb.train(params, dtrain, num_boost_round=5000, evals=[(dvalid, "valid")],
                          early_stopping_rounds=200, verbose_eval=False)
        t1 = time.time()
        lam_hat = model.predict(dall, iteration_range=(0, model.best_iteration+1)).clip(min=1e-6)
        print(f"[XGBoost-CPU] trained in {t1 - t0:.1f}s, best_iter={model.best_iteration}")
        return lam_hat.astype(np.float64), "XGBoost-Poisson-CPU"
    except Exception as e:
        raise RuntimeError(f"All GBM backends failed: {e}")

# CITY AGGREGATION
def aggregate_city_exact(df_in: pd.DataFrame, lam_col: str) -> pd.DataFrame:
    out = (df_in.loc[df_in[lam_col] >= LAMBDA_THRESH]
              .groupby(DATE_COL, as_index=False)[lam_col].sum()
              .rename(columns={lam_col: "lambda_city"}))
    try:
        from scipy.stats import poisson
        out["expected_total"] = out["lambda_city"]
        out["q05"] = poisson.ppf(0.05, out["lambda_city"]).astype(float)
        out["q50"] = poisson.ppf(0.50, out["lambda_city"]).astype(float)
        out["q95"] = poisson.ppf(0.95, out["lambda_city"]).astype(float)
    except Exception:
        z05, z95 = -1.644853626951, 1.644853626951
        mu = out["lambda_city"].values
        sd = np.sqrt(mu)
        out["expected_total"] = mu
        out["q05"] = np.clip(mu + z05*sd, 0, None)
        out["q50"] = mu
        out["q95"] = mu + z95*sd
    return out.sort_values("date").reset_index(drop=True)

def aggregate_city_mc(df_in: pd.DataFrame, lam_col: str, S=MC_SAMPLES) -> pd.DataFrame:
    """CPU (numpy) MC sum; use only if you really need sampling bands."""
    rows = []
    for date, sub in df_in.groupby(DATE_COL, sort=True):
        lam = sub.loc[sub[lam_col] >= LAMBDA_THRESH, lam_col].astype(np.float64).values
        if lam.size == 0:
            rows.append((date, 0.0, 0.0, 0.0, 0.0, 0))
            continue
        sims = np.random.poisson(lam, size=(S, lam.size)).sum(axis=1)
        rows.append((date, lam.sum(), sims.mean(), np.quantile(sims, 0.05), np.quantile(sims, 0.5), np.quantile(sims, 0.95), lam.size))
    return pd.DataFrame(rows, columns=["date","expected_total","sim_mean","sim_q05","sim_q50","sim_q95","active_boxes"]).sort_values("date")

def aggregate_city_mc_gpu(df_in: pd.DataFrame, lam_col: str, S=MC_SAMPLES) -> pd.DataFrame:
    """GPU MC using PyTorch if available; otherwise falls back to CPU MC."""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return aggregate_city_mc(df_in, lam_col, S=S)

    rows = []
    for date, sub in df_in.groupby(DATE_COL, sort=True):
        lam = sub.loc[sub[lam_col] >= LAMBDA_THRESH, lam_col].astype(np.float32).values
        if lam.size == 0:
            rows.append((date, 0.0, 0.0, 0.0, 0.0, 0))
            continue
        rate = torch.tensor(lam, device=device)
        sims = torch.poisson(rate.repeat(S, 1)).sum(dim=1).double().cpu().numpy()
        rows.append((date, lam.sum(), sims.mean(), np.quantile(sims, 0.05), np.quantile(sims, 0.5), np.quantile(sims, 0.95), lam.size))
    return pd.DataFrame(rows, columns=["date","expected_total","sim_mean","sim_q05","sim_q50","sim_q95","active_boxes"]).sort_values("date")

# MAIN
def main():
    print("Loading enriched dataset…")
    gdf = gpd.read_parquet(PARQUET)
    df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

    # Build features (full dataset, to avoid leakage in lags)
    print("Building features…")
    df, X_cols = build_features(df)

    # Split labeled training window
    train_mask_time = pd.to_datetime(df[DATE_COL]) <= pd.Timestamp("2023-12-31")
    train = df.loc[train_mask_time & df[Y_COL].notna()].copy()
    if train.empty:
        raise ValueError("No labeled rows in training window.")
    print(f"Train rows: {len(train):,}")

    # Design matrices
    Xtr = sanitize_X(train[X_cols].values)
    ytr = train[Y_COL].values.astype(np.float32)
    assert np.isfinite(ytr).all() and (ytr >= 0).all(), "Training target must be finite and >=0."

    Xall = sanitize_X(df[X_cols].values)

    # Fit & predict
    print("Training boosted Poisson model (GPU-first)…")
    lam_all, model_name = fit_predict_gbm(Xtr, ytr, Xall, train, X_cols)
    df[f"lam_{model_name}"] = lam_all

    # Optional per-cell export
    if OUT_PER_CELL:
        out_cells = df[[DATE_COL, ID_COL, Y_COL, f"lam_{model_name}"]].copy()
        out_cells.to_csv(OUT_PER_CELL, index=False)
        print(f"Wrote per-cell predictions → {OUT_PER_CELL}")

    # City-level aggregation
    print("Aggregating to city level…")
    if USE_EXACT_AGG:
        city = aggregate_city_exact(df, f"lam_{model_name}")
        # for consistency with MC output names:
        city = city.rename(columns={"lambda_city":"expected_total","q05":"sim_q05","q50":"sim_q50","q95":"sim_q95"})
        # number of active boxes
        active = (df.loc[df[f"lam_{model_name}"] >= LAMBDA_THRESH]
                    .groupby(DATE_COL, as_index=False)[f"lam_{model_name}"].size()
                    .rename(columns={"size":"active_boxes"}))
        city = city.merge(active, on="date", how="left")
        city["sim_mean"] = city["expected_total"]
    else:
        # Monte-Carlo (GPU torch if available)
        if MC_USE_TORCH:
            city = aggregate_city_mc_gpu(df, f"lam_{model_name}", S=MC_SAMPLES)
        else:
            city = aggregate_city_mc(df, f"lam_{model_name}", S=MC_SAMPLES)

    # Attach city truth (sum over boxes where available)
    city_truth = (df.loc[df[Y_COL].notna()]
                    .groupby(DATE_COL, as_index=False)[Y_COL].sum()
                    .rename(columns={Y_COL: "city_truth"}))
    city = city.merge(city_truth, on="date", how="left")
    city["model"] = model_name

    # City-level metrics (over dates with truth)
    eval_mask = city["city_truth"].notna()
    y_true = city.loc[eval_mask, "city_truth"].values.astype(float)
    y_hat  = city.loc[eval_mask, "sim_mean"].values.astype(float)
    metrics = eval_all(y_true, y_hat)
    print("\n=== City-level metrics ===")
    print({"model": model_name, **metrics})

    # Save
    city.to_csv(OUT_CITY, index=False)
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(OUT_SUMMARY, index=False)
    print(f"\nWrote city daily predictions → {OUT_CITY}")
    print(f"Wrote metric summary → {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
