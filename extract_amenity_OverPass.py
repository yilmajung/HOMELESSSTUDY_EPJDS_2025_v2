import requests
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "SF-OSM-amenities-historical/1.0 (contact: wjung@psu.edu)"
}

def get_area_id(place_query="San Francisco, California, USA"):
    """
    Resolve the OSM relation id using Nominatim and convert to Overpass 'area' id.
    (We only need this once; Overpass will time-travel the data via the [date:...] clause.)
    """
    params = {
        "q": place_query,
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
        "polygon_geojson": 0,
    }
    resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"No Nominatim result for: {place_query}")

    osm_type = results[0].get("osm_type")
    osm_id = int(results[0].get("osm_id"))

    if osm_type not in {"relation", "way", "node"}:
        raise ValueError(f"Unexpected osm_type from Nominatim: {osm_type}")

    return 3600000000 + osm_id

def build_overpass_query(area_id, snapshot_iso):
    """
    Build a historical Overpass QL query at the given snapshot time (UTC ISO string).
    We include:
      - amenity={restaurant, school, college, university, shelter, social_facility}
      - social_facility=shelter (homeless shelters)
      - ways/relations with bridge=yes
      - highway=*_link (ramps)
    We ask for 'center' so ways/relations get point centroids for easy mapping.
    """
    amenity_set = [
        "restaurant", "school", "college", "university",
        "shelter", "social_facility"
    ]
    amenity_regex = "|".join(amenity_set)
    ramps_regex = ".*_link"

    query = f"""
    [out:json][timeout:180][date:"{snapshot_iso}"];
    area({area_id})->.searchArea;
    (
      // amenities (incl. social_facility as amenity)
      node["amenity"~"^{amenity_regex}$"](area.searchArea);
      way["amenity"~"^{amenity_regex}$"](area.searchArea);
      relation["amenity"~"^{amenity_regex}$"](area.searchArea);

      // explicit homeless shelters
      node["amenity"="social_facility"]["social_facility"="shelter"](area.searchArea);
      way["amenity"="social_facility"]["social_facility"="shelter"](area.searchArea);
      relation["amenity"="social_facility"]["social_facility"="shelter"](area.searchArea);

      // bridges
      way["bridge"="yes"](area.searchArea);
      relation["bridge"="yes"](area.searchArea);

      // highway ramps (links)
      way["highway"~"{ramps_regex}"](area.searchArea);
      relation["highway"~"{ramps_regex}"](area.searchArea);
    );
    out tags center;
    """
    return query

def run_overpass(query, max_tries=4, backoff=12):
    """
    Execute Overpass query with gentle retry/backoff for rate limits / transient errors.
    """
    for attempt in range(1, max_tries + 1):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=300)
            # Handle rate limiting explicitly
            if resp.status_code in (429, 504) or "Too Many Requests" in resp.text:
                time.sleep(backoff * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == max_tries:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("Failed to fetch Overpass data after retries.")

def to_geodataframe(overpass_json, snapshot_date_str):
    """
    Convert Overpass JSON to GeoDataFrame (point geometries).
    Adds:
      - snapshot_date (YYYY-MM-DD)
      - osm_type/osm_id/tags
      - feature_type convenience label
    """
    elements = overpass_json.get("elements", [])
    recs = []
    for el in elements:
        osm_type = el.get("type")
        osm_id   = el.get("id")
        tags     = el.get("tags", {}) or {}

        # Simple labeling
        if "amenity" in tags:
            if tags.get("amenity") == "social_facility" and tags.get("social_facility") == "shelter":
                feature_type = "shelter"
            else:
                feature_type = f"amenity:{tags.get('amenity')}"
        elif tags.get("social_facility") == "shelter":
            feature_type = "shelter"
        elif tags.get("bridge") == "yes":
            feature_type = "bridge"
        elif "highway" in tags and str(tags.get("highway","")).endswith("_link"):
            feature_type = "highway_link"
        else:
            feature_type = "other"

        # Geometry as point
        if osm_type == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        else:
            center = el.get("center")
            lat = center.get("lat") if center else None
            lon = center.get("lon") if center else None

        if lat is None or lon is None:
            continue

        recs.append({
            "snapshot_date": snapshot_date_str,
            "osm_type": osm_type,
            "osm_id": osm_id,
            "name": tags.get("name"),
            "feature_type": feature_type,
            "amenity": tags.get("amenity"),
            "social_facility": tags.get("social_facility"),
            "highway": tags.get("highway"),
            "bridge": tags.get("bridge"),
            "tags": tags,
            "lat": lat,
            "lon": lon,
        })

    df = pd.DataFrame(recs)
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326"
    )
    return gdf

def fetch_sf_amenities_by_year(
    place="San Francisco, California, USA",
    save_per_year=False,
    per_year_basename="sf_osm_{year}.geojson",
    save_combined_geojson=None,
    save_combined_csv=None,
    polite_pause=6
):
    """
    For each year, query snapshot:
      - 2016..2023 at Dec 31 (23:59:59Z)
      - 2024 at May 31 (23:59:59Z)
    Returns a combined GeoDataFrame. Optionally saves per-year and/or combined outputs.
    """
    # Build the date list you requested
    dates = []
    for year in range(2016, 2024):
        dates.append(f"{year}-12-31")
    dates.append("2024-05-31")

    area_id = get_area_id(place)

    all_gdfs = []
    for d in dates:
        # Use end-of-day UTC to include the full date’s edits
        snap_iso = f"{d}T23:59:59Z"
        q = build_overpass_query(area_id, snap_iso)
        data = run_overpass(q)
        gdf = to_geodataframe(data, snapshot_date_str=d)
        all_gdfs.append(gdf)

        if save_per_year:
            # One file per year/date — include the date in filename for clarity
            year = d[:4]
            out_path = per_year_basename.format(year=year)
            # If you want unique filenames per exact date, do:
            # out_path = per_year_basename.format(year=d)
            if not gdf.empty:
                gdf.to_file(out_path, driver="GeoJSON")
            else:
                # Write an empty placeholder GeoJSON if desired; otherwise skip
                pass

        # Be polite to Overpass
        time.sleep(polite_pause)

    if len(all_gdfs) == 0:
        return gpd.GeoDataFrame(columns=["snapshot_date"], geometry=[], crs="EPSG:4326")

    combined = pd.concat(all_gdfs, ignore_index=True)
    combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")

    # Optional combined saves
    if save_combined_geojson:
        combined_gdf.to_file(save_combined_geojson, driver="GeoJSON")
    if save_combined_csv:
        cols = [c for c in combined_gdf.columns if c != "geometry"]
        combined_gdf[cols].to_csv(save_combined_csv, index=False)

    return combined_gdf

if __name__ == "__main__":
    # Example usage:
    gdf_all = fetch_sf_amenities_by_year(
        place="San Francisco, California, USA",
        save_per_year=True,                         # set to False if you don't want per-year files
        per_year_basename="sf_osm_amenities_{year}.geojson",
        save_combined_geojson="sf_osm_amenities_2016_2024.geojson",
        save_combined_csv="sf_osm_amenities_2016_2024.csv",
        polite_pause=8                              # increase if you hit rate limits
    )
    print(f"Total features across snapshots: {len(gdf_all)}")
    print(gdf_all.groupby('snapshot_date')['osm_id'].count())
    print(gdf_all.head())