import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

# Config
MAIN_CSV_PATH   = "~/HOMELESSSTUDY_EPJDS_2025_v2/df_311_cleaned_20250902.csv"
MAIN_DATE_COL   = "timestamp"
MAIN_GEOM_COL   = "geometry_bbox"
MAIN_GRID_IDCOL = "bbox_id"

AMENITIES_CSV   = "data/sf_osm_amenities_2016_2024.csv"   # from the Overpass step
OUTPUT_PATH     = "data/311_daily_with_amenities.parquet"

# Which amenity buckets to attach
CATS = [
    "restaurant", "school", "college", "university", "fast_food", "bank", "atm", "place_of_worship", "bench", "police", "cinema",
    "shelter_homeless", "shelter_generic",
    "bridge", "highway_link",
]

# Helpers
def load_main_grid_from_csv(csv_path, date_col, wkt_col, grid_id_col):
    """Load daily CSV with WKT polygons -> GeoDataFrame (EPSG:4326).
       Ensures date dtype, repair invalid polygons, ensure a grid_id."""
    # Try to read dates efficiently
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in {csv_path}")

    # Accept common alternates for the geometry column
    if wkt_col not in df.columns:
        for alt in ["geometry_bbox", "wkt", "geom"]:
            if alt in df.columns:
                wkt_col = alt
                break
        else:
            raise ValueError(f"WKT geometry column '{wkt_col}' not found. "
                             f"Available columns: {list(df.columns)[:20]} ...")

    # Parse dates (assume naive local or UTC-agnostic)
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    # Parse WKT -> geometry
    geom = gpd.GeoSeries.from_wkt(df[wkt_col])
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")

    # Fix invalid polygons (e.g., self-intersections)
    # Only buffer invalid ones to avoid unnecessary work
    gdf["geometry"] = gdf["geometry"].apply(lambda g: g.buffer(0) if (g is not None and not g.is_valid) else g)
    # Drop empties if any
    gdf = gdf[~gdf.geometry.is_empty].copy()

    # Ensure grid_id
    if grid_id_col not in gdf.columns:
        b = gdf.geometry.bounds.round(6)
        gdf["grid_id"] = (
            b["minx"].astype(str) + "_" + b["miny"].astype(str) + "_" +
            b["maxx"].astype(str) + "_" + b["maxy"].astype(str)
        )
        grid_id_col = "grid_id"

    return gdf, grid_id_col

def classify_category(row) -> str:
    a  = row.get("amenity", "")
    sf = row.get("social_facility", "")
    hw = row.get("highway", "")
    br = row.get("bridge", "")

    # Homeless shelters (explicit)
    if a == "social_facility" and sf == "shelter":
        return "shelter_homeless"
    # Generic shelters (bus/picnic)
    if a == "shelter":
        return "shelter_generic"
    # Core amenities
    if a in {"restaurant", "school", "college", "university", "fast_food", "bank", "atm", "place_of_worship", "bench", "police", "cinema"}:
        return a
    # Infrastructure
    if br in {"yes", "true", "1"}:
        return "bridge"
    if isinstance(hw, str) and hw.endswith("_link"):
        return "highway_link"
    return "other"


def year_to_snapshot(d: pd.Timestamp) -> pd.Timestamp:
    """Map any daily date to snapshot (EOY for 2016–2023; 2024-05-31)."""
    y = d.year
    if y < 2016 or (y == 2024 and d > pd.Timestamp("2024-05-31")) or y > 2024:
        raise ValueError(f"Date {d.date()} outside supported range 2016-01-01..2024-05-31.")
    return pd.Timestamp("2024-05-31") if y == 2024 else pd.Timestamp(f"{y}-12-31")


# Load main daily grid (CSV+WKT)
main_gdf, GRID_ID_COL = load_main_grid_from_csv(
    MAIN_CSV_PATH, MAIN_DATE_COL, MAIN_GEOM_COL, MAIN_GRID_IDCOL
)

# Unique grid for spatial joins
grid_unique = main_gdf[[GRID_ID_COL, "geometry"]].drop_duplicates(subset=[GRID_ID_COL]).reset_index(drop=True)
if grid_unique.crs is None:
    grid_unique.set_crs("EPSG:4326", inplace=True)

# Load OSM amenities
amen = pd.read_csv(AMENITIES_CSV)
need = {"snapshot_date", "lat", "lon", "amenity", "social_facility", "highway", "bridge", "osm_type", "osm_id"}
missing = need - set(amen.columns)
if missing:
    raise ValueError(f"Amenities CSV is missing columns: {missing}")

amen["snapshot_date"] = pd.to_datetime(amen["snapshot_date"])
amen = amen.drop_duplicates(subset=["snapshot_date", "osm_type", "osm_id"], keep="first").reset_index(drop=True)

amen_gdf = gpd.GeoDataFrame(
    amen,
    geometry=gpd.points_from_xy(amen["lon"], amen["lat"]),
    crs="EPSG:4326",
)
if amen_gdf.crs != grid_unique.crs:
    amen_gdf = amen_gdf.to_crs(grid_unique.crs)

# Make sure these columns exist and are lowercase strings (no NaNs)
for col in ["amenity", "social_facility", "highway", "bridge"]:
    if col not in amen_gdf.columns:
        amen_gdf[col] = pd.Series(pd.NA, index=amen_gdf.index)
    amen_gdf[col] = amen_gdf[col].astype("string").str.lower().fillna("")

amen_gdf["category"] = amen_gdf.apply(classify_category, axis=1)
snapshots = amen_gdf["snapshot_date"].drop_duplicates().sort_values().tolist()

# Count amenities per grid per snapshot
per_snap = []
for snap in snapshots:
    pts = amen_gdf.loc[amen_gdf["snapshot_date"] == snap, ["category", "geometry"]].copy()
    if pts.empty:
        continue

    joined = gpd.sjoin(
        pts,
        grid_unique[[GRID_ID_COL, "geometry"]],
        how="inner",
        predicate="within",
    )

    counts = (
        joined.groupby([GRID_ID_COL, "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATS, fill_value=0)       # ensure all buckets exist
        .reset_index()
        .rename(columns={c: f"n_{c}" for c in CATS})
    )
    counts["snapshot_date"] = pd.to_datetime(snap)
    per_snap.append(counts)

if per_snap:
    snap_counts = pd.concat(per_snap, ignore_index=True)
else:
    # Empty shell: all zeros for expected snapshots
    snap_counts = grid_unique[[GRID_ID_COL]].copy()
    for c in CATS:
        snap_counts[f"n_{c}"] = 0
    expected = [pd.Timestamp(f"{y}-12-31") for y in range(2016, 2024)] + [pd.Timestamp("2024-05-31")]
    snap_counts = snap_counts.assign(key=1).merge(
        pd.DataFrame({"snapshot_date": expected, "key": 1}), on="key", how="outer"
    ).drop(columns="key").fillna(0)

# Optional total (amenities only; exclude infra if you want)
core = ["n_restaurant", "n_school", "n_college", "n_university", "n_fast_food", "n_bank", "n_atm", "n_place_of_worship", "n_bench", "n_police", "n_cinema", "n_shelter_homeless", "n_shelter_generic"]
for c in core:
    if c not in snap_counts.columns:
        snap_counts[c] = 0
snap_counts["n_amenities_total"] = snap_counts[core].sum(axis=1)


# =========================
# Map each daily row to its snapshot & merge
# =========================
main = main_gdf.copy()
main["snapshot_date"] = main[MAIN_DATE_COL].apply(year_to_snapshot)

merged = main.merge(
    snap_counts,
    on=[GRID_ID_COL, "snapshot_date"],
    how="left"
)

# Fill missing counts
for c in [f"n_{x}" for x in CATS] + ["n_amenities_total"]:
    if c not in merged.columns:
        merged[c] = 0
    merged[c] = merged[c].fillna(0).astype("Int64")

# =========================
# Save
# =========================
ext = Path(OUTPUT_PATH).suffix.lower()
if ext in [".parquet"]:
    merged.to_parquet(OUTPUT_PATH, index=False)
elif ext in [".feather", ".arrow"]:
    merged.to_feather(OUTPUT_PATH)
elif ext in [".csv"]:
    out = merged.copy()
    # Write geometry back to WKT so it stays portable in CSV
    out["geometry_wkt"] = out.geometry.to_wkt()
    out = out.drop(columns="geometry")
    out.to_csv(OUTPUT_PATH, index=False)
elif ext in [".geojson"]:
    merged.to_file(OUTPUT_PATH, driver="GeoJSON")
elif ext in [".gpkg"]:
    merged.to_file(OUTPUT_PATH, layer="daily", driver="GPKG")
else:
    # default to Parquet if unknown extension
    merged.to_parquet(OUTPUT_PATH if OUTPUT_PATH.endswith(".parquet") else OUTPUT_PATH + ".parquet", index=False)

print("Done. Added columns:")
print([f"n_{x}" for x in CATS] + ["n_amenities_total"])