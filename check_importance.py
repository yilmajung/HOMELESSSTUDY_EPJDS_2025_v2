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
scaler = joblib.load('scaler_pois_t400r300_withamenities_parquet.joblib')
constant_mean = joblib.load('constant_mean_t400r300_withamenities_parquet.pkl')
inducing_points = torch.load('inducing_points_pois_t400r300_withamenities_parquet.pt', map_location=device)

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
        #mean_x = mean_x.clamp(min=-3.0, max=3.0) # No clamping
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
ENRICHED_PARQUET = "data/main_daily_with_amenities.parquet"

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

model.load_state_dict(torch.load('stvgp_pois_t400r300_withamenities_parquet.pth', map_location=device))
likelihood.load_state_dict(torch.load('likelihood_pois_t400r300_withamenities_parquet.pth', map_location=device))


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

# Check importance
# ARD-based importance over the covariate kernel (magnitude only)
ls = model.covariate_kernel.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)  # shape [d_cov]
score = 1.0 / (ls**2 + 1e-12)
ard_df = pd.DataFrame({
    "feature": X_cols,
    "lengthscale": ls,
    "importance_1_over_ls2": score,
    "importance_norm": score / score.sum()
}).sort_values("importance_norm", ascending=False).reset_index(drop=True)

print("\nTop ARD features (shorter lengthscale ⇒ stronger variation along that feature):")
print(ard_df.head(15))
print(ard_df.tail(15))

# Gradient-based direction (sign), averaged across a sample
# Sample a manageable subset
rng = np.random.default_rng(0)
N = min(20000, test_x_np.shape[0])
idx = rng.choice(test_x_np.shape[0], size=N, replace=False)
X_unscaled = test_x_np[idx].astype(np.float32)

# Scale again (so we can track scales)
X_scaled = scaler.transform(X_unscaled).astype(np.float32)

bs = 1024
cov_start = 3  # [lat, lon]=0:2, time=2, covariates start at 3
grads_sum = np.zeros(len(X_cols), dtype=np.float64)
pos_frac  = np.zeros(len(X_cols), dtype=np.float64)
count     = 0

model.eval()
with torch.no_grad():  # we'll re-enable grad per-batch
    pass

for i in tqdm(range(0, N, bs), desc="Gradients"):
    xb = torch.tensor(X_scaled[i:i+bs], device=device, dtype=torch.float32, requires_grad=True)
    with torch.enable_grad():
        post = model(xb)
        m = post.mean  # shape [bs]
        # total gradient of sum(m) wrt inputs
        grad = torch.autograd.grad(outputs=m.sum(), inputs=xb, retain_graph=False, create_graph=False)[0]
        g_cov = grad[:, cov_start:]  # keep covariate part
        grads_sum += g_cov.detach().cpu().numpy().sum(axis=0)
        pos_frac  += (g_cov.detach().cpu().numpy() > 0).sum(axis=0)
        count     += g_cov.shape[0]

# Average gradient in scaled space
avg_grad_scaled = grads_sum / max(count, 1)

# Convert to original units (chain rule: dx_scaled/dx = 1/std)
stds = np.array(scaler.scale_, dtype=np.float64)[cov_start:]
avg_grad_original_units = avg_grad_scaled / stds

dir_df = pd.DataFrame({
    "feature": X_cols,
    "avg_grad_scaled": avg_grad_scaled,
    "avg_grad_per_unit_in_original": avg_grad_original_units,
    "frac_positive_grad": pos_frac / max(count, 1)
}).sort_values("avg_grad_per_unit_in_original", ascending=False).reset_index(drop=True)

print("\nGradient-based direction summary (positive values ⇒ increasing feature raises log-rate):")
print(dir_df.head(15))

# Save ard- and gradient-based direction summary
ard_df.to_csv('data/ard_importance_summary.csv', index=False)
dir_df.to_csv('data/gradient_based_direction_summary.csv', index=False)
