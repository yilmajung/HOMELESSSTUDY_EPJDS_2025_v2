#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dallas tents pipeline (IDs + shapefile → GroundingDINO → YOLO-classifier → export)

Install:
    pip install requests pillow numpy opencv-python tqdm ultralytics groundingdino \
                shapely geopandas pyproj fiona python-dotenv

Env:
    export MAPILLARY_TOKEN=YOUR_TOKEN
"""

import argparse
import csv
import glob
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm

# Geo
import geopandas as gpd
from shapely.geometry import Point

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MAPILLARY_BASE = "https://graph.mapillary.com"
REQ_TIMEOUT = 20
RETRY_MAX = 5

GD_CAPTION = "tent"
GD_BOX_THRESH = 0.4
GD_TEXT_THRESH = 0.4

YOLO_POS_THRESHOLD = 0.50
TENT_CLASS_NAME = "tent"

DB_DEFAULT = "tent_pipeline.sqlite"
OUT_DIR_DEFAULT = "outputs"
CACHE_DIR_DEFAULT = "image_cache"


@dataclass
class Paths:
    db: str
    out_dir: str
    cache_dir: str


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    lon REAL,
    lat REAL,
    captured_at TEXT,
    in_dallas INTEGER,      -- 0/1 after shapefile filter
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS stage1 (
    image_id TEXT PRIMARY KEY,
    has_tent_guess INTEGER,
    boxes_json TEXT,
    processed_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS stage2 (
    image_id TEXT PRIMARY KEY,
    is_tent INTEGER,
    prob REAL,
    details_json TEXT,
    processed_at TEXT,
    error TEXT
);

CREATE VIEW IF NOT EXISTS positives AS
SELECT i.image_id, i.lon, i.lat, i.captured_at, s2.prob
FROM images i
JOIN stage1 s1 ON i.image_id = s1.image_id
JOIN stage2 s2 ON i.image_id = s2.image_id
WHERE i.in_dallas = 1 AND s1.has_tent_guess = 1 AND s2.is_tent = 1;
"""


def ensure_db(db_path: str):
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA_SQL)
    con.commit()
    con.close()


def db_conn(db_path: str):
    return sqlite3.connect(db_path, timeout=60)


def mk_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"OAuth {token}"}


def request_with_retries(url: str, headers: Dict[str, str], params: Dict[str, str], desc: str):
    backoff = 1.6
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((backoff ** attempt))
                continue
            raise RuntimeError(f"{desc}: HTTP {r.status_code} - {r.text[:300]}")
        except requests.RequestException as e:
            if attempt == RETRY_MAX:
                raise RuntimeError(f"{desc}: {e}") from e
            time.sleep((backoff ** attempt))
    raise RuntimeError(f"{desc}: retry limit exceeded")


def fetch_image_meta(token: str, image_id: str):
    headers = mk_headers(token)
    fields = "id,geometry,captured_at"
    url = f"{MAPILLARY_BASE}/{image_id}"
    data = request_with_retries(url, headers, {"fields": fields}, f"Get meta {image_id}")
    coords = data.get("geometry", {}).get("coordinates", [None, None])
    lon, lat = (coords[0], coords[1]) if coords and len(coords) == 2 else (None, None)
    captured_at = data.get("captured_at")
    return lon, lat, captured_at


def fetch_image_bytes(token: str, image_id: str, prefer_1024=True) -> bytes:
    headers = mk_headers(token)
    fields = "thumb_1024_url" if prefer_1024 else "thumb_2048_url"
    url = f"{MAPILLARY_BASE}/{image_id}"
    resp = request_with_retries(url, headers, {"fields": fields}, f"Get thumb for {image_id}")
    key = "thumb_1024_url" if prefer_1024 else "thumb_2048_url"
    thumb_url = resp.get(key)
    if not thumb_url:
        raise RuntimeError(f"No thumb url for {image_id}")
    r = requests.get(thumb_url, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.content


def maybe_cache_image(paths: Paths, image_id: str, img_bytes: bytes) -> str:
    Path(paths.cache_dir).mkdir(parents=True, exist_ok=True)
    out = Path(paths.cache_dir) / f"{image_id}.jpg"
    if not out.exists():
        with open(out, "wb") as f:
            f.write(img_bytes)
    return str(out)


# -----------------------
# Ingest helpers
# -----------------------

def _iso_from_ms(ms_str: Optional[str]) -> Optional[str]:
    if not ms_str:
        return None
    try:
        ms = int(ms_str)
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _read_input_csvs(csv_or_glob: str) -> List[dict]:
    files = []
    if any(c in csv_or_glob for c in ["*", "?", "["]):
        files = sorted(glob.glob(csv_or_glob))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {csv_or_glob}")
    else:
        if not os.path.exists(csv_or_glob):
            raise FileNotFoundError(csv_or_glob)
        files = [csv_or_glob]

    rows: List[dict] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            needed = {"id", "captured_at_ms", "lon", "lat", "url"}
            if not needed.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{fp} must have headers: {', '.join(sorted(needed))}")
            for r in reader:
                rows.append({
                    "image_id": r["id"].strip(),
                    "lon": float(r["lon"]) if r.get("lon") else None,
                    "lat": float(r["lat"]) if r.get("lat") else None,
                    "captured_at": _iso_from_ms(r.get("captured_at_ms")),
                    # r["url"] is intentionally ignored (may be expired)
                })
    return rows


def ingest_csv(paths: Paths, csv_or_glob: str, shapefile: str, token: Optional[str]):
    ensure_db(paths.db)
    rows = _read_input_csvs(csv_or_glob)

    # If lon/lat missing, enrich via Mapillary (needs token)
    need_geo = [r for r in rows if r["lon"] is None or r["lat"] is None]
    if need_geo:
        if not token:
            print("Some rows are missing lon/lat; please provide --mapillary-token to enrich.", file=sys.stderr)
            sys.exit(2)
        print(f"Enriching {len(need_geo)} rows with lon/lat from Mapillary...")
        for r in tqdm(need_geo, desc="Enrich meta"):
            lon, lat, cap = fetch_image_meta(token, r["image_id"])
            r["lon"], r["lat"] = lon, lat
            if r["captured_at"] is None:
                r["captured_at"] = cap

    # Load Dallas polygon
    gdf_city = gpd.read_file(shapefile)
    if gdf_city.crs is None:
        gdf_city.set_crs(epsg=4326, inplace=True)
    else:
        gdf_city = gdf_city.to_crs(epsg=4326)

    # Build point GeoDataFrame
    pts = []
    for r in rows:
        if r["lon"] is not None and r["lat"] is not None:
            pts.append(Point(r["lon"], r["lat"]))
        else:
            pts.append(None)
    gdf = gpd.GeoDataFrame(rows, geometry=pts, crs="EPSG:4326")

    # Spatial filter (point-in-polygon)
    city_union = gdf_city.unary_union
    in_flags = []
    for geom in gdf.geometry:
        if geom is None:
            in_flags.append(0)
        else:
            # contains or touches city boundary
            in_flags.append(1 if (city_union.contains(geom) or city_union.intersects(geom)) else 0)
    gdf["in_dallas"] = in_flags

    # Insert into DB
    with db_conn(paths.db) as con:
        cur = con.cursor()
        for _, r in gdf.iterrows():
            cur.execute(
                "INSERT OR REPLACE INTO images(image_id, lon, lat, captured_at, in_dallas, fetched_at) VALUES(?,?,?,?,?,datetime('now'))",
                (r["image_id"], r["lon"], r["lat"], r["captured_at"], int(r["in_dallas"]))
            )
        con.commit()

    kept = int(gdf["in_dallas"].sum())
    print(f"Ingested {len(gdf)} images from {csv_or_glob}; {kept} inside Dallas polygon.")


# -----------------------
# GroundingDINO (Stage 1)
# -----------------------

def load_groundingdino(config_path: str, weights_path: str):
    from groundingdino.util.inference import load_model
    return load_model(config_path, weights_path)

def gdin_predict(model, pil_img: Image.Image, caption: str, box_thresh: float, text_thresh: float):
    from groundingdino.util.inference import predict
    import groundingdino.datasets.transforms as T
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(pil_img, None)
    img = img_tensor.unsqueeze(0)
    boxes, logits, phrases = predict(model, img, caption, box_thresh, text_thresh)
    out = []
    for b, s, ph in zip(boxes, logits, phrases):
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        out.append([x1, y1, x2, y2, float(s), str(ph)])
    return out

def _shard_filter(ids: List[str], num_shards: int, shard_index: int) -> List[str]:
    if num_shards <= 1:
        return ids
    sel = []
    for x in ids:
        # stable shard assignment based on Python's hash is process-randomized across runs;
        # use a stable hash instead:
        h = sum(ord(c) for c in x) % num_shards
        if h == shard_index:
            sel.append(x)
    return sel

def run_stage1(paths: Paths, token: str, config_path: str, weights_path: str,
               limit: int = 0, use_cache=True, num_shards: int = 1, shard_index: int = 0):
    ensure_db(paths.db)
    model = load_groundingdino(config_path, weights_path)

    with db_conn(paths.db) as con:
        cur = con.cursor()
        # Only images inside Dallas, not yet processed in stage1
        cur.execute("""
            SELECT image_id FROM images
            WHERE in_dallas = 1 AND image_id NOT IN (SELECT image_id FROM stage1)
        """)
        ids = [r[0] for r in cur.fetchall()]

    ids = _shard_filter(ids, num_shards, shard_index)
    if limit > 0:
        ids = ids[:limit]

    pbar = tqdm(ids, desc=f"Stage1 GDINO [shard {shard_index}/{num_shards}]")
    with db_conn(paths.db) as con:
        cur = con.cursor()
        for image_id in pbar:
            try:
                img_bytes = fetch_image_bytes(token, image_id)
                if use_cache:
                    maybe_cache_image(paths, image_id, img_bytes)
                pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                dets = gdin_predict(model, pil, GD_CAPTION, GD_BOX_THRESH, GD_TEXT_THRESH)
                has_guess = 1 if len(dets) > 0 else 0
                cur.execute(
                    "INSERT OR REPLACE INTO stage1(image_id, has_tent_guess, boxes_json, processed_at, error) VALUES (?,?,?,?,NULL)",
                    (image_id, has_guess, json.dumps(dets), time.strftime("%Y-%m-%d %H:%M:%S"))
                )
            except Exception as e:
                cur.execute(
                    "INSERT OR REPLACE INTO stage1(image_id, has_tent_guess, boxes_json, processed_at, error) VALUES (?,?,?,?,?)",
                    (image_id, None, None, time.strftime("%Y-%m-%d %H:%M:%S"), str(e)[:500])
                )
            con.commit()


# -----------------------
# YOLO Classification (Stage 2)
# -----------------------

def load_yolo_cls(weights_path: str):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    return model

def yolo_classify_tent_prob(model, pil_img: Image.Image) -> Tuple[float, Dict]:
    results = model(pil_img, verbose=False)
    res = results[0]
    probs = getattr(res, "probs", None)
    if probs is None or not hasattr(probs, "data"):
        raise RuntimeError("Provided YOLO weights don't look like a *classification* model.")

    names = results.names if hasattr(results, "names") else getattr(model, "names", {})
    tent_idx = None
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == TENT_CLASS_NAME:
                tent_idx = int(k)
                break
    if tent_idx is None:
        tent_idx = 0  # fallback for single-class models

    vec = probs.data.cpu().numpy().ravel()
    tent_prob = float(vec[tent_idx]) if tent_idx < len(vec) else 0.0
    details = {"names": names, "top1": int(getattr(probs, "top1", -1))}
    return tent_prob, details

def run_stage2(paths: Paths, token: str, yolo_weights: str,
               limit: int = 0, use_cache=True, num_shards: int = 1, shard_index: int = 0):
    ensure_db(paths.db)
    model = load_yolo_cls(yolo_weights)

    with db_conn(paths.db) as con:
        cur = con.cursor()
        # only those flagged by stage1 and missing in stage2
        cur.execute("""
            SELECT s1.image_id
            FROM stage1 s1
            LEFT JOIN stage2 s2 ON s1.image_id = s2.image_id
            JOIN images i ON i.image_id = s1.image_id
            WHERE s1.has_tent_guess = 1 AND s2.image_id IS NULL AND i.in_dallas = 1
        """)
        ids = [r[0] for r in cur.fetchall()]

    ids = _shard_filter(ids, num_shards, shard_index)
    if limit > 0:
        ids = ids[:limit]

    pbar = tqdm(ids, desc=f"Stage2 YOLO-cls [shard {shard_index}/{num_shards}]")
    with db_conn(paths.db) as con:
        cur = con.cursor()
        for image_id in pbar:
            try:
                img_bytes = fetch_image_bytes(token, image_id)
                if use_cache:
                    maybe_cache_image(paths, image_id, img_bytes)
                pil = Image.open(BytesIO(img_bytes)).convert("RGB")
                tent_prob, details = yolo_classify_tent_prob(model, pil)
                is_tent = 1 if tent_prob >= YOLO_POS_THRESHOLD else 0
                cur.execute(
                    "INSERT OR REPLACE INTO stage2(image_id, is_tent, prob, details_json, processed_at, error) VALUES (?,?,?,?,?,NULL)",
                    (image_id, is_tent, tent_prob, json.dumps(details), time.strftime("%Y-%m-%d %H:%M:%S"))
                )
            except Exception as e:
                cur.execute(
                    "INSERT OR REPLACE INTO stage2(image_id, is_tent, prob, details_json, processed_at, error) VALUES (?,?,?,?,?,?)",
                    (image_id, None, None, None, time.strftime("%Y-%m-%d %H:%M:%S"), str(e)[:500])
                )
            con.commit()


# -----------------------
# Export
# -----------------------

def export_positives(paths: Paths, out_csv: Optional[str] = None):
    ensure_db(paths.db)
    Path(paths.out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = out_csv or str(Path(paths.out_dir) / "tent_positives.csv")
    with db_conn(paths.db) as con, open(out_csv, "w", encoding="utf-8") as f:
        cur = con.cursor()
        cur.execute("""
            SELECT i.image_id, i.lon, i.lat, i.captured_at, s2.prob
            FROM positives p
            JOIN images i ON p.image_id = i.image_id
            JOIN stage2 s2 ON p.image_id = s2.image_id
        """)
        f.write("image_id,lon,lat,captured_at,prob\n")
        for row in cur:
            image_id, lon, lat, cap, prob = row
            f.write(f"{image_id},{lon},{lat},{cap},{prob}\n")
    print(f"Exported positives → {out_csv}")


# -----------------------
# CLI
# -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dallas tent pipeline (CSV(s) + shapefile → GDINO → YOLO-cls)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p.add_argument("--db", default=DB_DEFAULT)
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)

    # ingest
    i = sub.add_parser("ingest", help="Load CSV(s) and filter to Dallas shapefile")
    i.add_argument("--csv", required=True, help="CSV path or glob (e.g., /data/dallas_mapillary_image_data.csv or /data/chunk_*.csv)")
    i.add_argument("--shapefile", required=True, help="Path to Dallas shapefile (.shp)")
    i.add_argument("--mapillary-token", default=os.getenv("MAPILLARY_TOKEN"))

    # stage1
    s1 = sub.add_parser("stage1", help="Run GroundingDINO prefilter")
    s1.add_argument("--mapillary-token", default=os.getenv("MAPILLARY_TOKEN"))
    s1.add_argument("--grounding-config", required=True)
    s1.add_argument("--grounding-weights", required=True)
    s1.add_argument("--limit", type=int, default=0)
    s1.add_argument("--no-cache", action="store_true")
    s1.add_argument("--num-shards", type=int, default=1)
    s1.add_argument("--shard-index", type=int, default=0)

    # stage2 (YOLO classification)
    s2 = sub.add_parser("stage2", help="Run YOLO classification confirm")
    s2.add_argument("--mapillary-token", default=os.getenv("MAPILLARY_TOKEN"))
    s2.add_argument("--yolo-weights", required=True)
    s2.add_argument("--limit", type=int, default=0)
    s2.add_argument("--no-cache", action="store_true")
    s2.add_argument("--num-shards", type=int, default=1)
    s2.add_argument("--shard-index", type=int, default=0)

    # export
    e = sub.add_parser("export", help="Export final positives to CSV")
    e.add_argument("--out-csv", default=None)

    # all-in-one
    a = sub.add_parser("all", help="Run: ingest -> stage1 -> stage2 -> export")
    a.add_argument("--csv", required=True)
    a.add_argument("--shapefile", required=True)
    a.add_argument("--mapillary-token", default=os.getenv("MAPILLARY_TOKEN"))
    a.add_argument("--grounding-config", required=True)
    a.add_argument("--grounding-weights", required=True)
    a.add_argument("--yolo-weights", required=True)
    a.add_argument("--limit-s1", type=int, default=0)
    a.add_argument("--limit-s2", type=int, default=0)
    a.add_argument("--no-cache", action="store_true")
    a.add_argument("--out-csv", default=None)
    a.add_argument("--num-shards", type=int, default=1)
    a.add_argument("--shard-index", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    paths = Paths(db=args.db, out_dir=args.out_dir, cache_dir=args.cache_dir)
    Path(paths.out_dir).mkdir(parents=True, exist_ok=True)
    Path(paths.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.cmd == "ingest":
        if not args.mapillary_token:
            print("NOTE: If any CSV rows lack lon/lat, provide --mapillary-token to enrich.", file=sys.stderr)
        ingest_csv(paths, args.csv, args.shapefile, args.mapillary_token)

    elif args.cmd == "stage1":
        if not args.mapillary_token:
            print("ERROR: Provide --mapillary-token (needed to fetch fresh image thumbs).", file=sys.stderr)
            sys.exit(2)
        run_stage1(paths, args.mapillary_token, args.grounding_config, args.grounding_weights,
                   limit=max(0, args.limit),
                   use_cache=(not args.no_cache),
                   num_shards=max(1, args.num_shards),
                   shard_index=max(0, args.shard_index))

    elif args.cmd == "stage2":
        if not args.mapillary_token:
            print("ERROR: Provide --mapillary-token (needed to fetch fresh image thumbs).", file=sys.stderr)
            sys.exit(2)
        run_stage2(paths, args.mapillary_token, args.yolo_weights,
                   limit=max(0, args.limit),
                   use_cache=(not args.no_cache),
                   num_shards=max(1, args.num_shards),
                   shard_index=max(0, args.shard_index))

    elif args.cmd == "export":
        export_positives(paths, args.out_csv)

    elif args.cmd == "all":
        if not args.mapillary_token:
            print("ERROR: Provide --mapillary-token", file=sys.stderr)
            sys.exit(2)
        ingest_csv(paths, args.csv, args.shapefile, args.mapillary_token)
        run_stage1(paths, args.mapillary_token, args.grounding_config, args.grounding_weights,
                   limit=max(0, args.limit_s1),
                   use_cache=(not args.no_cache),
                   num_shards=max(1, args.num_shards),
                   shard_index=max(0, args.shard_index))
        run_stage2(paths, args.mapillary_token, args.yolo_weights,
                   limit=max(0, args.limit_s2),
                   use_cache=(not args.no_cache),
                   num_shards=max(1, args.num_shards),
                   shard_index=max(0, args.shard_index))
        export_positives(paths, args.out_csv)

    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
