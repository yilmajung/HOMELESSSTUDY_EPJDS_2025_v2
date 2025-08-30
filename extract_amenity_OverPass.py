# Requires: pip install requests pandas geopandas shapely pyproj

import requests
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "SF-OSM-amenities-script/1.0 (contact: wjung@psu.edu)"
}

def get_area_id(place_query="San Francisco, California, USA"):
    """
    Use Nominatim to look up the OSM relation for 'place_query'
    and convert it to an Overpass 'area' id (relation_id + 3600000000).
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

    # Overpass 'area' ids are:
    # - relation: 3600000000 + osm_id
    # - way:      3600000000 + osm_id  (not typical for admin boundaries)
    # - node:     3600000000 + osm_id  (rare for areas)
    if osm_type == "relation":
        area_id = 3600000000 + osm_id
    elif osm_type == "way":
        area_id = 3600000000 + osm_id
    elif osm_type == "node":
        area_id = 3600000000 + osm_id
    else:
        raise ValueError(f"Unexpected osm_type from Nominatim: {osm_type}")

    return area_id

def build_overpass_query(area_id):
    """
    Build a single Overpass QL query that fetches:
      - amenity = restaurant, school (and higher ed), shelter variants
      - social_facility=shelter (often how homeless shelters are tagged)
      - ways/relations with bridge=yes
      - highway link/ramps (motorway_link, trunk_link, primary_link, etc.)
    We ask Overpass to provide 'center' for ways/relations so we can map them as points.
    """
    # Amenity patterns (keep simple & explicit)
    amenity_set = [
        "restaurant", "fast_food", "school", "college", "university",
        "shelter", "social_facility"
    ]

    amenity_regex = "|".join(amenity_set)

    # Highway ramps: usually *_link (e.g., motorway_link, trunk_link, primary_link)
    ramps_regex = ".*_link"

    # Query:
    # - Pull nodes/ways/relations for amenities
    # - Specifically pull the homeless-shelter schema via social_facility=shelter
    # - Pull bridges (bridge=yes) from ways & relations
    # - Pull highway links (ramps)
    # - 'out tags center;' gives tags and a computed center for ways/relations
    query = f"""
    [out:json][timeout:180];
    area({area_id})->.searchArea;
    (
      // General amenities (including 'social_facility' as an amenity tag)
      node["amenity"~"^{amenity_regex}$"](area.searchArea);
      way["amenity"~"^{amenity_regex}$"](area.searchArea);
      relation["amenity"~"^{amenity_regex}$"](area.searchArea);

      // Homeless shelters tagging pattern:
      node["amenity"="social_facility"]["social_facility"="shelter"](area.searchArea);
      way["amenity"="social_facility"]["social_facility"="shelter"](area.searchArea);
      relation["amenity"="social_facility"]["social_facility"="shelter"](area.searchArea);

      // Bridges
      way["bridge"="yes"](area.searchArea);
      relation["bridge"="yes"](area.searchArea);

      // Highway ramps
      way["highway"~"{ramps_regex}"](area.searchArea);
      relation["highway"~"{ramps_regex}"](area.searchArea);
    );
    out tags center;
    """
    return query

def run_overpass(query, max_tries=3, backoff=10):
    """
    Run the Overpass query with light retry/backoff for courtesy.
    """
    for attempt in range(1, max_tries + 1):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=300)
            if resp.status_code == 429 or "Too Many Requests" in resp.text:
                # Rate-limited; wait and retry
                time.sleep(backoff * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == max_tries:
                raise
            time.sleep(backoff * attempt)
    raise RuntimeError("Failed to fetch Overpass data after retries.")

def to_geodataframe(overpass_json):
    """
    Convert Overpass JSON to a GeoDataFrame with point geometries.
    - For nodes: use lat/lon
    - For ways/relations: use 'center' (provided by 'out center')
    Adds:
      - osm_type (node/way/relation)
      - osm_id
      - feature_type (amenity/bridge/highway_link...)
    """
    elements = overpass_json.get("elements", [])
    recs = []
    for el in elements:
        osm_type = el.get("type")
        osm_id   = el.get("id")
        tags     = el.get("tags", {}) or {}

        # Determine a simple 'feature_type' label for convenience
        feature_type = None
        if "amenity" in tags:
            # prioritize 'social_facility=shelter' labeling when applicable
            if tags.get("amenity") == "social_facility" and tags.get("social_facility") == "shelter":
                feature_type = "shelter"
            else:
                feature_type = f"amenity:{tags.get('amenity')}"
        elif tags.get("social_facility") == "shelter":
            feature_type = "shelter"
        elif tags.get("bridge") == "yes":
            feature_type = "bridge"
        elif "highway" in tags and tags.get("highway", "").endswith("_link"):
            feature_type = "highway_link"
        else:
            feature_type = "other"

        # Geometry
        if osm_type == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        else:
            center = el.get("center")
            lat = center.get("lat") if center else None
            lon = center.get("lon") if center else None

        if lat is None or lon is None:
            # Occasionally, an element might be missing center; skip to keep things simple
            continue

        rec = {
            "osm_type": osm_type,
            "osm_id": osm_id,
            "name": tags.get("name"),
            "feature_type": feature_type,
            "amenity": tags.get("amenity"),
            "social_facility": tags.get("social_facility"),
            "highway": tags.get("highway"),
            "bridge": tags.get("bridge"),
            "tags": tags,   # keep full tag dict for reference
            "lat": lat,
            "lon": lon,
        }
        recs.append(rec)

    df = pd.DataFrame(recs)
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326"
    )
    return gdf

def fetch_sf_amenities(place="San Francisco, California, USA", save_geojson=None, save_csv=None):
    """
    End-to-end convenience function:
      - resolve area id
      - query Overpass
      - return GeoDataFrame (WGS84)
      - optionally save to GeoJSON/CSV
    """
    area_id = get_area_id(place)
    query = build_overpass_query(area_id)
    data = run_overpass(query)
    gdf = to_geodataframe(data)

    # Optional saves
    if save_geojson:
        gdf.to_file(save_geojson, driver="GeoJSON")
    if save_csv:
        # For CSV, drop the geometry or split lon/lat
        cols = [c for c in gdf.columns if c != "geometry"]
        gdf[cols].to_csv(save_csv, index=False)

    return gdf

if __name__ == "__main__":
    gdf = fetch_sf_amenities(
        place="San Francisco, California, USA",
        save_geojson="data/sf_osm_amenities_links_bridges.geojson",
        save_csv="data/sf_osm_amenities_links_bridges.csv"
    )
    print(f"Fetched {len(gdf)} features.")
    print(gdf.head(5))
