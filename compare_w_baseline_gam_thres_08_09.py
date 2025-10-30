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
from scipy.special import gammaln

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
P_THRESH = np.arange(0.3, 0.92, 0.02).tolist() # thresholds to test
LAMBDA_THRESH = [-math.log(1.0 - p) for p in P_THRESH]

# Outputs
OUT_DAILY   = "city_daily_predictions_baselines_gam_only.csv"
OUT_SUMMARY = "baseline_city_metrics_gam_only.csv"

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

# Load ground-truth sf data
df_sf = pd.read_csv('sf_tent.csv')

# Drop rows with NaN in 'date' column
df_sf = df_sf.dropna(subset=['date'], axis=0)

# Create timestamp column in df_sf using year, month, and day
df_sf.rename(columns={'date': 'day'}, inplace=True)
df_sf['timestamp'] = pd.to_datetime(df_sf[['year', 'month', 'day']])

# Ensure we have a 'timestamp' (datetime) column aligned with df_sf
if 'timestamp' not in df_all.columns:
    df_all['timestamp'] = pd.to_datetime(df_all[DATE_COL])
else:
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])


# Basic settings
S = 500
z95 = 1.96
EPS = 1e-12

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


# ================================
# Baseline 2b: Negative Binomial & Zero-Inflated models (statsmodels)
# ================================
print("Training NB / ZIP baselines…")
from threadpoolctl import threadpool_limits

# statsmodels imports
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP

# Ensure integer counts for SM discrete models
y_tr_int = np.asarray(y_tr, dtype=np.int64)

# Add intercept (statsmodels needs explicit constant)
X_tr_sm  = sm.add_constant(X_tr_sc, has_constant='add')
X_all_sm = sm.add_constant(X_all_sc, has_constant='add')

lam_nb = None
lam_zip = None
lam_zinb = None

# ---- NB (discrete NB2; estimates dispersion) ----
try:
    with threadpool_limits(limits=1):
        nb_mod = sm.NegativeBinomial(endog=y_tr_int, exog=X_tr_sm, loglike_method='nb2')  # NB2 variance
        nb_fit = nb_mod.fit(method='lbfgs', disp=False, maxiter=300)
    lam_nb = np.clip(nb_fit.predict(X_all_sm), 1e-6, None)
    df_all["lam_nb"] = lam_nb
    print(" NB baseline trained.")
except Exception as e:
    print(f" NB baseline failed: {e}")

# ---- ZIP (zero-inflated Poisson with logit inflation) ----
# Inflation design: keep it simple & stable → use just a constant and lag7 (if present)
infl_cols = []
if "lag7" in df_all.columns:
    # IMPORTANT: use the **scaled** version for stability; we can reuse GLM scaling
    # Build an inflation array aligned to X_all_sc:
    # we don't have a scaler per-column names, so directly create an array:
    # constant + standardized lag7
    # Pull raw lag7 then z-score on train
    lag7_tr = train["lag7"].to_numpy(dtype=float)
    lag7_all = df_all["lag7"].to_numpy(dtype=float)
    lag7_mu, lag7_sd = lag7_tr.mean(), lag7_tr.std() if lag7_tr.std() > 0 else 1.0
    infl_tr = (lag7_tr - lag7_mu) / lag7_sd
    infl_all = (lag7_all - lag7_mu) / lag7_sd

    infl_exog_tr  = np.column_stack([np.ones_like(infl_tr), infl_tr]).astype(np.float32)
    infl_exog_all = np.column_stack([np.ones_like(infl_all), infl_all]).astype(np.float32)
else:
    infl_exog_tr  = np.ones((len(y_tr_int), 1), dtype=np.float32)   # constant-only inflation
    infl_exog_all = np.ones((len(df_all), 1), dtype=np.float32)

try:
    with threadpool_limits(limits=1):
        zip_mod = ZeroInflatedPoisson(
            endog=y_tr_int,
            exog=X_tr_sm,
            exog_infl=infl_exog_tr,
            inflation='logit'
        )
        zip_fit = zip_mod.fit(method='lbfgs', disp=False, maxiter=300)
    lam_zip = np.clip(zip_fit.predict(exog=X_all_sm, exog_infl=infl_exog_all, which='mean'), 1e-6, None)
    df_all["lam_zip"] = lam_zip
    print(" ZIP baseline trained.")
except Exception as e:
    print(f" ZIP baseline failed: {e}")

# ---- ZINB (optional; more flexible than ZIP if data is strongly overdispersed) ----
try:
    with threadpool_limits(limits=1):
        zinb_mod = ZeroInflatedNegativeBinomialP(
            endog=y_tr_int,
            exog=X_tr_sm,
            exog_infl=infl_exog_tr,
            inflation='logit'
        )
        zinb_fit = zinb_mod.fit(method='lbfgs', disp=False, maxiter=300)
    lam_zinb = np.clip(zinb_fit.predict(exog=X_all_sm, exog_infl=infl_exog_all, which='mean'), 1e-6, None)
    df_all["lam_zinb"] = lam_zinb
    print(" ZINB baseline trained.")
except Exception as e:
    print(f" ZINB baseline failed: {e}")




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
# === Robustified GAM-ST ===
from pygam import PoissonGAM, s, te
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

def fit_predict_gam_st_robust(gdf, df, date_col="date", y_col="ground_truth",
                              use_pseudo_grid_re=False):
    df = df.copy()
    # Ensure lon/lat
    if not {"latitude","longitude"}.issubset(df.columns):
        cent = gdf.geometry.centroid
        df["latitude"]  = cent.y.values
        df["longitude"] = cent.x.values

    # Time index
    t0 = pd.to_datetime(df[date_col]).min()
    df["t_idx"] = (pd.to_datetime(df[date_col]) - t0).dt.days.astype(int)

    # Cycles (already in your code, but ensure existence)
    if "dow" not in df.columns:   df["dow"]   = pd.to_datetime(df[date_col]).dt.weekday
    if "month" not in df.columns: df["month"] = pd.to_datetime(df[date_col]).dt.month
    for newc, arr in {
        "dow_sin":   np.sin(2*np.pi*df["dow"]/7.0),
        "dow_cos":   np.cos(2*np.pi*df["dow"]/7.0),
        "month_sin": np.sin(2*np.pi*df["month"]/12.0),
        "month_cos": np.cos(2*np.pi*df["month"]/12.0),
    }.items():
        if newc not in df.columns: df[newc] = arr

    base_covs  = ["max","min","precipitation","total_population","white_ratio","black_ratio","hh_median_income"]
    amen_feats = [c for c in df.columns if c.startswith("log1p_n_")]
    lag_feats  = [c for c in df.columns if c in ("lag7","lag28")]
    cyc_feats  = ["dow_sin","dow_cos","month_sin","month_cos"]

    X_cols = ["longitude","latitude","t_idx"] + cyc_feats + base_covs + amen_feats + lag_feats

    # Z-score for stability
    X_mat = df[X_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    scaler = StandardScaler().fit(X_mat)
    Xz = scaler.transform(X_mat).astype(np.float32)
    y  = df[y_col].fillna(0).values.astype(np.float32)  # unlabeled rows as 0; they won't be used in training

    # Train mask
    train_mask = (pd.to_datetime(df[date_col]) <= pd.Timestamp("2023-12-31")) & df[y_col].notna()
    Xtr, ytr = Xz[train_mask], y[train_mask]

    # Build terms:
    # modest spatial surface + modest time smooth; keep others near-linear (2 knots)
    nsp_space = 20  # less flexible than 30
    nsp_time  = 15
    terms = te(0,1, n_splines=nsp_space, spline_order=3) + s(2, n_splines=nsp_time)

    # pseudo-random effect: penalized spline on a numeric grid index (optional)
    if use_pseudo_grid_re:
        # Map grid id to 0..G-1
        grid_codes = pd.factorize(df[ID_COL])[0].astype(np.int32)
        df["_grid_code"] = grid_codes
        X_cols.append("_grid_code")
        # Append to Xz
        Xz = np.column_stack([Xz, grid_codes.astype(np.float32)])
        Xtr = Xz[train_mask]
        # Strong penalty, order 1 (ridge-like)
        terms += s(len(X_cols)-1, n_splines= df[ID_COL].nunique()//50 + 2, spline_order=1)

    # Time-blocked CV for smoothing params
    tscv = TimeSeriesSplit(n_splits=5)
    lam_grid = np.logspace(0, 4, 7)  # stronger penalties than default
    gam = PoissonGAM(terms, fit_intercept=True, max_iter=2000)
    gam = gam.gridsearch(Xtr, ytr, lam=lam_grid, cv=tscv, progress=False)

    # Predict mean rate
    lam_hat = np.clip(gam.predict_mu(Xz), 1e-6, None)
    return lam_hat, "GAM_ST_robust"


print("Training Spatiotemporal Poisson GAM (GAM-ST)…")
try:
    lam_gam, gam_name = fit_predict_gam_st_robust(gdf, df_all, date_col=DATE_COL, y_col=Y_COL,
                                                  use_pseudo_grid_re=False)
    df_all["lam_gam_st"] = lam_gam
except Exception as e:
    print(f"GAM-ST robust version skipped: {e}")
    gam_name = None


##############################################
# City-level MC aggregation in STVGP style
##############################################

def mean_poisson_deviance(y, mu, eps=EPS):
    y  = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return 2.0 * np.mean(y * (np.log((y + eps) / (mu + eps))) - (y - mu))

def log_pois_pmf(k, lam, eps=EPS):
    return k*np.log(lam + eps) - lam - gammaln(k + 1.0)

def day_log_predictive(k, lambda_draws):
    lp = log_pois_pmf(k, lambda_draws)
    m  = np.max(lp)
    return m + np.log(np.mean(np.exp(lp - m) + 0.0))

def stvgp_style_city_eval(df_cells, lam_col, df_sf, p_thresh):
    """
    df_cells: per-cell dataframe with columns [timestamp, <lam_col>]
    lam_col : column with per-cell mean λ̂
    p_thresh: threshold on P(Y>0) = 1 - exp(-median)
    Returns: (df_daily, metrics_dict, lambda_draws_by_day)
    """
    df_combined = df_cells[['timestamp', lam_col]].copy()
    df_combined = df_combined.rename(columns={lam_col: 'median'})  # use λ̂ as 'median'
    # no process uncertainty for baselines → set lower95 = median (σ=0)
    df_combined['lower95'] = df_combined['median']

    daily_out  = []
    lambda_draws_by_day = {}
    n_days = df_combined['timestamp'].nunique()

    for day, grp in df_combined.groupby('timestamp', sort=True):
        med = grp['median'].values.astype(float)
        l95 = grp['lower95'].values.astype(float)

        # P(Y_i>0) = 1 - exp(-median)
        p_any = 1.0 - np.exp(-med)
        keep_mask = p_any >= p_thresh

        if not keep_mask.any():
            daily_out.append({
                'timestamp':  day,
                'mean_total': 0.0,
                'median_total': 0.0,
                'lower95': 0.0,
                'upper95': 0.0,
                'lower90': 0.0,
                'upper90': 0.0,
                'active_boxes': 0,
            })
            # also store degenerate Λ draws (=0)
            lambda_draws_by_day[day] = np.zeros(S, dtype=float)
            continue

        med_filt = med[keep_mask]
        l95_filt = l95[keep_mask]
        nbox     = med_filt.size

        # Reconstruct Normal(f) params from (median, l95)
        # For baselines: l95 == median → sigma = 0 → degenerate
        mu_f    = np.log(np.maximum(med_filt, EPS))
        sigma_f = (np.log(np.maximum(med_filt, EPS)) - np.log(np.maximum(l95_filt, EPS))) / z95
        sigma_f = np.clip(sigma_f, 0.0, None)

        # Monte Carlo draws of log-rate f, then λ = exp(f)
        f_samps = np.random.normal(loc=mu_f[None, :], scale=sigma_f[None, :], size=(S, nbox))
        lam_samps = np.exp(f_samps)
        lam_samps = np.minimum(lam_samps, 1e3)  # same optional cap

        # Poisson sampling of counts per box, sum to city
        y_samps    = np.random.poisson(lam_samps)
        city_samps = y_samps.sum(axis=1)

        # Store Λ_s (sum of λ_s across boxes) for mixture-of-Poissons NLPD
        lambda_draws_by_day[day] = lam_samps.sum(axis=1)

        daily_out.append({
            'timestamp':   day,
            'mean_total':  city_samps.mean(),
            'median_total': np.median(city_samps),
            'lower95':     np.percentile(city_samps, 2.5),
            'upper95':     np.percentile(city_samps, 97.5),
            'lower90':     np.percentile(city_samps, 5.0),
            'upper90':     np.percentile(city_samps, 95.0),
            'active_boxes': int(nbox),
        })

    df_daily = pd.DataFrame(daily_out).sort_values('timestamp').reset_index(drop=True)

    # Align with ground-truth df_sf (your exact filter)
    df_eval = (
        df_sf[df_sf['timestamp'] < '2024-01-01'][['timestamp','tent_struc']]
          .merge(df_daily[['timestamp','mean_total']], on='timestamp', how='inner')
          .rename(columns={'tent_struc':'y','mean_total':'yhat'})
          .sort_values('timestamp')
          .reset_index(drop=True)
    )
    y    = df_eval['y'].to_numpy(dtype=float)
    yhat = df_eval['yhat'].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    mape = float(np.mean(np.abs(yhat - y) / (y + EPS)) * 100.0)
    mpd  = float(mean_poisson_deviance(y, yhat))

    # Mixture-of-Poissons NLPD
    logps = []
    for ts, k in zip(df_eval['timestamp'].tolist(), y.astype(int)):
        lam_draws = lambda_draws_by_day.get(ts, None)
        if lam_draws is None or len(lam_draws) == 0:
            lam_draws = np.array([df_eval.loc[df_eval['timestamp']==ts, 'yhat'].item()], dtype=float)
        logps.append(day_log_predictive(k, lam_draws))
    nlpd = float(-np.mean(logps))

    metrics = {"RMSE": rmse, "MAPE(%)": mape, "MeanPoissonDev": mpd, "NLPD": nlpd}
    return df_daily, metrics, lambda_draws_by_day

# -------- run for multiple thresholds and models you care about --------
# Create a P_LIST from 0.2 to 0.9 with step 0.02
P_LIST = np.arange(0.3, 0.92, 0.02).tolist() # thresholds to test

MODELS_FOR_SWEEP = [
    ("lam_seasonal7",   "SeasonalNaive_lag7"),
    ("lam_poisson_glm", "PoissonGLM_L2"),
    ("lam_gbm",         gbm_name)       if "lam_gbm" in df_all else None,
    ("lam_gam_st",      "GAM_ST")       if "lam_gam_st" in df_all else None,
    ("lam_nb",          "NB2_discrete") if "lam_nb" in df_all else None,
    ("lam_zip",         "ZIP_logit")    if "lam_zip" in df_all else None,
    ("lam_zinb",        "ZINB_logit")   if "lam_zinb" in df_all else None,
]
MODELS_FOR_SWEEP = [m for m in MODELS_FOR_SWEEP if m is not None]


all_daily, all_summary = [], []
for lam_col, model_name in MODELS_FOR_SWEEP:
    for p in P_LIST:
        daily_df, metrics, _ = stvgp_style_city_eval(df_all[['timestamp', lam_col]].copy(),
                                                     lam_col=lam_col, df_sf=df_sf, p_thresh=p)
        daily_df["model"] = model_name
        daily_df["p_thresh"] = p
        all_daily.append(daily_df)

        row = {"model": model_name, "p_thresh": p, **metrics}
        all_summary.append(row)

df_daily_sweep  = pd.concat(all_daily, ignore_index=True).sort_values(["model","p_thresh","timestamp"])
df_summary_sweep = pd.DataFrame(all_summary).sort_values(["model","p_thresh"]).reset_index(drop=True)

print("\n=== STVGP-style threshold sweep summary ===")
print(df_summary_sweep.head(20))

# Optional: save
df_daily_sweep.to_csv("gam_final2_city_daily_predictions_threshold_sweep_stvgpstyle.csv", index=False)
df_summary_sweep.to_csv("gam_final2_city_metrics_threshold_sweep_stvgpstyle.csv", index=False)