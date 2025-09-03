import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import torch
import gpytorch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import re
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn.functional as F
from gpytorch.mlls import VariationalELBO
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.likelihoods import _OneDimensionalLikelihood

ENRICHED_PARQUET = "data/mapillary_daily_with_amenities.parquet"

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


# Filter to training rows (with ground truth)
df_training = gdf[~gdf["ground_truth"].isna()].copy()
if df_training.empty:
    raise ValueError("No rows with ground_truth found after filtering.")

# Base covariates (fill if missing)
base_covs = ["max", "min", "precipitation", "total_population", "white_ratio", "black_ratio", "hh_median_income"]
for c in base_covs:
    if c not in df_training.columns:
        df_training[c] = 0.0

# Amenity count columns: all n_* except an optional total to avoid double counting
amen_cols_all = [c for c in df_training.columns if c.startswith("n_")]
amen_cols = [c for c in amen_cols_all if c != "n_amenities_total"]
if not amen_cols:
    raise ValueError("No amenity count columns (n_*) found in the enriched file.")

# log1p transform to reduce skew
for c in amen_cols:
    df_training[c] = pd.to_numeric(df_training[c], errors="coerce").fillna(0)
    df_training[f"log1p_{c}"] = np.log1p(df_training[c].astype(float))

amen_feat_cols = [f"log1p_{c}" for c in amen_cols]
X_cols = base_covs + amen_feat_cols

# Extract arrays for GP
spatial_coords = df_training[['latitude', 'longitude']].values
temporal_coords = df_training[['timestamp']].values
X_covariates = df_training[X_cols].astype(np.float32).values
y_counts = df_training['ground_truth'].values

# Inducing points selection (use bboxid if available, else grid_id)
print("Selecting inducing points…")
cell_id_col = "bboxid"

num_density_points = 400
num_random_points  = 300

bbox_counts = df_training.groupby(cell_id_col)["ground_truth"].mean().reset_index()
top_density_ids = bbox_counts.nlargest(num_density_points, "ground_truth")[cell_id_col].values
remaining      = bbox_counts[~bbox_counts[cell_id_col].isin(top_density_ids)]
random_ids     = remaining.sample(n=min(num_random_points, len(remaining)), random_state=42)[cell_id_col].values

inducing_ids = np.unique(np.concatenate([top_density_ids, random_ids]))
inducing_df  = df_training[df_training[cell_id_col].isin(inducing_ids)].drop_duplicates(subset=[cell_id_col])

# Build inducing arrays
Z_spatial = inducing_df[['latitude','longitude']].values
Z_temporal = inducing_df[['timestamp']].values
Z_covariates = inducing_df[X_cols].astype(np.float32).values


# Sanity checks
for arr_name, arr in [("spatial_coords", spatial_coords), ("temporal_coords", temporal_coords),
                      ("X_covariates", X_covariates), ("Z_spatial", Z_spatial),
                      ("Z_temporal", Z_temporal), ("Z_covariates", Z_covariates)]:
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError(f"{arr_name} contains NaN/Inf. Check inputs.")

# Stack & scale
print("Scaling features…")
t = np.hstack((spatial_coords, temporal_coords, X_covariates)).astype(np.float32)
y = y_counts.astype(np.float32)

scaler = StandardScaler().fit(t)
x_scaled = scaler.transform(t).astype(np.float32)
joblib.dump(scaler, "scaler_ab_study_w_mapillary.joblib")

log_y_mean = np.log(y.mean() + 1e-3).astype(np.float32)
joblib.dump(log_y_mean, "constant_mean_ab_study_w_mapillary.pkl")

train_x = torch.tensor(x_scaled, dtype=torch.float32)
train_y = torch.tensor(y, dtype=torch.float32)

Z_np = np.hstack((Z_spatial, Z_temporal, Z_covariates)).astype(np.float32)
Z_scaled = scaler.transform(Z_np).astype(np.float32)
inducing_points = torch.tensor(Z_scaled, dtype=torch.float32)

# DataLoader (unchanged)
train_ds = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

print(f"Prepared {len(train_y)} training rows with {X_covariates.shape[1]} covariates "
      f"(base={len(base_covs)}, amenities={len(amen_feat_cols)}).")


# Define STVGP with ConstantMean
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
        self.covariate_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=X_covariates.shape[1]))
        self.const_kernel     = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.ConstantKernel())

    def forward(self, x):
        s, t, c = x[:, :2], x[:, 2:3], x[:, 3:]
        # Constant mean uses covariates to determine batch shape
        mean_x = self.mean_module(c)
        #mean_x = mean_x.clamp(min=-3.0, max=3.0) # remove clamping to see more variances
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


# class PoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
#     def __init__(self):
#         super().__init__()
#         # No parameters for vanilla Poisson

#     def forward(self, function_samples, **kwargs):
#         # The function_samples should be on log-scale
#         rate = function_samples.exp()
#         rate = torch.nan_to_num(rate, nan=1e-6, posinf=1e6, neginf=1e-6)
#         rate = rate.clamp(min=1e-6, max=1e6)  # Ensure rate is positive
#         return torch.distributions.Poisson(rate)
    
#     def expected_log_prob(self, target, function_dist, **kwargs):
#         mean = function_dist.mean
#         rate = mean.exp()
#         dist = torch.distributions.Poisson(rate)
#         return dist.log_prob(target)


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = STVGPModel(inducing_points.to(device), constant_mean=log_y_mean).to(device)
likelihood = QuadraturePoisson().to(device)

model.train()
likelihood.train()

# # ------- Variational posterior sanity check -------
# vd = model.variational_strategy._variational_distribution

# # 1) Mean
# print("variational_mean (first 5):", vd.variational_mean[:5])

# # 2) Scale‐tril diagonal
# print("scale_tril diag (first 5):", vd.chol_variational_covar.diag()[:5])

# # 3) List the raw parameters
# print("named_parameters under vd:")

# for name, param in vd.named_parameters():
#     print(f"  {name:30s} shape={tuple(param.shape)} requires_grad={param.requires_grad}")
# # --------------------------------------------------


# Optimizer & MLL
mll = VariationalELBO(likelihood, model, num_data=len(train_y))
optimizer = torch.optim.Adam(
    [
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01
)

for name, param in model.named_parameters():
    if "variational" in name:
        print(name, "requires_grad?", param.requires_grad)

# Training loop
print("Starting training...")

for epoch in tqdm(range(500)):
    total_loss = 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        output = model(x_b)
        loss = -mll(output, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss {total_loss:.3f}")
        vd = model.variational_strategy._variational_distribution
        print(f"\n=== After epoch {epoch} ===")
        print(" variational_mean[:5]:", vd.variational_mean[:5].cpu().detach().numpy())
        print(" scale_tril diag[:5]:", vd.chol_variational_covar.diag()[:5].cpu().detach().numpy())

# Save
torch.save(model.state_dict(), 'stvgp_ab_study_w_mapillary.pth')
torch.save(likelihood.state_dict(), 'likelihood_ab_study_w_mapillary.pth')
torch.save(inducing_points, 'inducing_points_ab_study_w_mapillary.pt')
