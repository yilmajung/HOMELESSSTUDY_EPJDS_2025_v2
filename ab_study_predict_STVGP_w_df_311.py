import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import torch.nn.functional as F
import re
from gpytorch.mlls import VariationalELBO
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.likelihoods import _OneDimensionalLikelihood
import geopandas as gpd

# Load the necessary files
print("Loading saved artifacts...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load('scaler_ab_study_w_311.joblib')
constant_mean = joblib.load('constant_mean_ab_study_w_311.pkl')
inducing_points = torch.load('inducing_points_ab_study_w_311.pt', map_location=device)

# Define the model and likelihood classes exactly as in training
# Define ST-VGP model

class STVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, constant_mean):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat= gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strat)

        self.mean_module = gpytorch.means.ConstantMean()
        # initialize constant to log mean of counts
        self.mean_module.constant.data.fill_(constant_mean)

        self.spatial_kernel   = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2))
        self.temporal_kernel  = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5))
        cov_dim = inducing_points.shape[1] - 3  # 2 spatial + 1 temporal
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=cov_dim))
        self.const_kernel     = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ConstantKernel())

    def forward(self, x):
        s, t, c = x[:, :2], x[:, 2:3], x[:, 3:]
        # Constant mean uses covariates to determine batch shape
        mean_x = self.mean_module(c)
        #mean_x = mean_x.clamp(min=-3.0, max=3.0)
        Ks = self.spatial_kernel(s)
        Kt = self.temporal_kernel(t)
        Kc = self.covariate_kernel(c)
        Kconst = self.const_kernel(s)

        covar = Ks * Kt * Kc + Ks + Kt + Kc + Kconst
        covar = covar + torch.eye(covar.size(-1), device=x.device) * 1e-3  # jitter
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class QuadraturePoisson(_OneDimensionalLikelihood):
    def __init__(self, num_locs=20):
        super().__init__()
        # Only pass the number of nodes
        self.quad = GaussHermiteQuadrature1D(num_locs)

    def forward(self, function_samples, **kwargs):
        rates = function_samples.exp().clamp(min=1e-6)
        return torch.distributions.Poisson(rates)

    def expected_log_prob(self, target, function_dist, **kwargs):
        # function_dist is the MultivariateNormal over f
        def log_prob_fn(f):
            # f has shape (num_locs, batch)
            # broadcast target → (num_locs, batch)
            return torch.distributions.Poisson(f.exp().clamp(min=1e-6)) \
                        .log_prob(target.unsqueeze(0))
        # Pass the *distribution* object, not mean/var
        return self.quad(log_prob_fn, function_dist)


# Load and preprocess the dataset (same as training)
ENRICHED_PARQUET = "data/311_daily_with_amenities.parquet"

print("Loading main dataset with amenities…")
gdf = gpd.read_parquet(ENRICHED_PARQUET)

# Basic checks
if "ground_truth" not in gdf.columns:
    raise ValueError("ground_truth not found in the enriched file; cannot train.")

# parse lat/lon, timestamp
print("Preprocessing dataset...")
gdf['latitude'] = gdf['center_latlon'].apply(lambda x: str(x.split(', ')[0]))
gdf['longitude'] = gdf['center_latlon'].apply(lambda x: str(x.split(', ')[1]))
gdf['latitude'] = gdf['latitude'].apply(lambda x: float(re.search(r'\d+.\d+', x).group()))
gdf['longitude'] = gdf['longitude'].apply(lambda x: float(re.search(r'\-\d+.\d+', x).group()))
gdf['timestamp'] = pd.to_datetime(gdf['timestamp'])
gdf['timestamp'] = (gdf['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

df_test = gdf[gdf['ground_truth'].isna()].copy()
# Sanity Check with training data
# print("Sanity check with training data...")
# df_test = df.dropna(subset=['ground_truth']) # actually this is the training data
# Small subset for testing
# df_test = df_test.sample(n=1000, random_state=42).reset_index(drop=True)


# Instantiate and load trained parameters
model = STVGPModel(inducing_points.to(device), constant_mean=constant_mean).to(device)
likelihood = QuadraturePoisson().to(device)

model.load_state_dict(torch.load('stvgp_ab_study_w_311.pth', map_location=device))
likelihood.load_state_dict(torch.load('likelihood_ab_study_w_311.pth', map_location=device))

# debugging lines
vd = model.variational_strategy._variational_distribution
print("variational-mean (first 5):", vd.variational_mean[:5])
# extract the Cholesky scale_tril diagonal:
scale_tril = vd.chol_variational_covar.diag()
print("variational-scale_tril diag (first 5):", scale_tril[:5])

model.eval()
likelihood.eval()

# Base covariates (fill if missing)
base_covs = ["max", "min", "precipitation", "total_population", "white_ratio", "black_ratio", "hh_median_income"]
for c in base_covs:
    if c not in df_test.columns:
        df_test[c] = 0.0

# Amenity count columns: all n_* except an optional total to avoid double counting
amen_cols_all = [c for c in df_test.columns if c.startswith("n_")]
amen_cols = [c for c in amen_cols_all if c != "n_amenities_total"]
if not amen_cols:
    raise ValueError("No amenity count columns (n_*) found in the enriched file.")

# log1p transform to reduce skew
for c in amen_cols:
    df_test[c] = pd.to_numeric(df_test[c], errors="coerce").fillna(0)
    df_test[f"log1p_{c}"] = np.log1p(df_test[c].astype(float))

amen_feat_cols = [f"log1p_{c}" for c in amen_cols]
X_cols = base_covs + amen_feat_cols


# Prepare test features
spatial_coords = df_test[['latitude', 'longitude']].values
temporal_coords = df_test[['timestamp']].values
X_covariates = df_test[X_cols].astype(np.float32).values

test_x_np = np.hstack((spatial_coords, temporal_coords, X_covariates))
test_x = torch.tensor(scaler.transform(test_x_np), dtype=torch.float32).to(device)

# Predict in batches
print("Predicting...")
batch_size = 512
num_lik_samples = 500

test_loader = DataLoader(TensorDataset(test_x), batch_size=batch_size, shuffle=False, drop_last=False)

# z‐score for 95% CI
z95 = 1.96
z90 = 1.645

# Containers for per‐box summaries
pred_mean_Y      = []
pred_var_Y       = []
pred_rate_median = []
pred_rate_lower95    = []
pred_rate_upper95    = []
pred_rate_lower90 = []
pred_rate_upper90 = []

# Per‐box predictive mean & variance of Y, plus log‐normal rate CI
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for (x_batch,) in tqdm(test_loader, desc="Per‐box stats"):
        post = model(x_batch)          # MultivariateNormal over f
        m = post.mean               # shape (bsz,)
        v = post.variance           # shape (bsz,)
        #print(f"  m → min {m.min():.2f}, max {m.max():.2f},  v → min {v.min():.2f}, max {v.max():.2f}")
        #m = m.clamp(min=-5.0, max=5.0)  
        #v = v.clamp(max=20.0)
        s = v.sqrt()                # σ_f

        # E[Y] = E[e^f] = exp(m + ½ v)
        E_Y    = torch.exp(m + 0.5 * v)
        exp_v_minus1 = torch.expm1(v)
        Var_e_f = E_Y * E_Y * exp_v_minus1
        Var_Y   = E_Y + Var_e_f

        # log‐normal rate summaries
        rate_med  = torch.exp(m)
        rate_l95b = torch.exp(m - z95 * s)
        rate_u95b = torch.exp(m + z95 * s)
        rate_l90b = torch.exp(m - z90 * s)
        rate_u90b = torch.exp(m + z90 * s)

        # append (move to CPU numpy)
        pred_mean_Y.append(E_Y.cpu().numpy())
        pred_var_Y.append(Var_Y.cpu().numpy())
        pred_rate_median.append(rate_med.cpu().numpy())
        pred_rate_lower95.append(rate_l95b.cpu().numpy())
        pred_rate_upper95.append(rate_u95b.cpu().numpy())
        pred_rate_lower90.append(rate_l90b.cpu().numpy())
        pred_rate_upper90.append(rate_u90b.cpu().numpy())

# flatten and attach to df_test
df_test = df_test.reset_index(drop=True)
df_test['pred_mean_Y']      = np.concatenate(pred_mean_Y)
df_test['pred_var_Y']       = np.concatenate(pred_var_Y)
df_test['rate_median']      = np.concatenate(pred_rate_median)
df_test['rate_lower95']     = np.concatenate(pred_rate_lower95)
df_test['rate_upper95']     = np.concatenate(pred_rate_upper95)
df_test['rate_lower90']     = np.concatenate(pred_rate_lower90)
df_test['rate_upper90']     = np.concatenate(pred_rate_upper90)

df_test.to_csv('stvgp_ab_study_w_311_noclamp.csv', index=False)
