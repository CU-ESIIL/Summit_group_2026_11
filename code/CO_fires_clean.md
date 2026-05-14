# Colorado Wildfire Prediction Pipeline
## Google Earth Engine + AlphaEarth Embeddings + XGBoost / Random Forest

**Study area:** Colorado  
**Fire data:** MTBS burned area boundaries  
**Predictors:** AlphaEarth satellite embeddings · TerraClimate/GRIDMET climate · SRTM topography  
**Models:** XGBoost classifier · Random Forest classifier · XGBoost area regressor  
**Output:** Fire probability and burned area predictions with spatial visualization

---

## Table of Contents

1. [Setup and Authentication](#1-setup-and-authentication)
2. [Explore MTBS Fire Data](#2-explore-mtbs-fire-data)
3. [Data Collection from GEE](#3-data-collection-from-gee)
   - 3a. [Fire perimeters + AEF embeddings + GRIDMET (polygon extraction)](#3a-fire-perimeters--aef-embeddings--gridmet-polygon-extraction)
   - 3b. [Centroid buffer extraction](#3b-centroid-buffer-extraction)
4. [Full Training Dataset — Fire / Non-fire Labels](#4-full-training-dataset--fire--non-fire-labels)
   - 4a. [Multi-year fire/non-fire with per-point climate and embeddings](#4a-multi-year-firenon-fire-with-per-point-climate-and-embeddings)
   - 4b. [2020 fires — nearby non-fire offset points](#4b-2020-fires--nearby-non-fire-offset-points)
   - 4c. [2020 fires — random non-fire points outside all MTBS burns](#4c-2020-fires--random-non-fire-points-outside-all-mtbs-burns)
5. [Export Shapefiles](#5-export-shapefiles)
6. [Final Training Dataset — Pre-fire predictors only](#6-final-training-dataset--pre-fire-predictors-only)
7. [XGBoost Classifier](#7-xgboost-classifier)
8. [Random Forest Classifier](#8-random-forest-classifier)
9. [XGBoost Area Regression (LOO-CV)](#9-xgboost-area-regression-loo-cv)
10. [Streamlit Digital Twin App](#10-streamlit-digital-twin-app)

---

## 1. Setup and Authentication

```python
import ee

# Authenticate once (opens browser)
ee.Authenticate()
ee.Initialize()
```

```bash
# Install required packages
pip install xgboost geemap geopandas
```

---

## 2. Explore MTBS Fire Data

Inspect the MTBS burned area boundary collection to understand its schema and filter to Colorado.

```python
import ee
ee.Initialize()

# --- Colorado boundary ---
states = ee.FeatureCollection("TIGER/2018/States")
colorado = states.filter(ee.Filter.eq("NAME", "Colorado")).geometry()

# --- MTBS collection ---
mtbs = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")

# Inspect schema
first = mtbs.first()
print("Property names:")
print(first.propertyNames().getInfo())

print("\nFirst feature:")
print(first.toDictionary().getInfo())

# Count fires in Colorado
print("Total MTBS fires in Colorado:")
print(mtbs.filterBounds(colorado).size().getInfo())
```

---

## 3. Data Collection from GEE

### 3a. Fire perimeters + AEF embeddings + GRIDMET (polygon extraction)

Extracts mean AlphaEarth embeddings (before/after/delta) and GRIDMET fire-weather
variables over MTBS fire polygons for 2020.

```python
import ee
import geemap
import pandas as pd

ee.Initialize()

# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
FIRE_YEAR    = 2020
BEFORE_YEAR  = 2019
AFTER_YEAR   = 2021
N_FIRES      = 20
OUT_CSV      = f"CO_{FIRE_YEAR}_MTBS_{N_FIRES}_AEF_GRIDMET_training_table.csv"

FIRE_DATE_FIELD = "Ig_Date"

# -------------------------------------------------------
# Study area
# -------------------------------------------------------
states   = ee.FeatureCollection("TIGER/2018/States")
colorado = states.filter(ee.Filter.eq("NAME", "Colorado")).geometry()

# -------------------------------------------------------
# MTBS — add fire year from Ig_Date
# -------------------------------------------------------
mtbs = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")

def add_fire_year_from_date(f):
    fire_date = ee.Date(f.get(FIRE_DATE_FIELD))
    return f.set({
        "fire_date":           fire_date.format("YYYY-MM-dd"),
        "fire_year_from_date": fire_date.get("year")
    })

mtbs_with_year = mtbs.map(add_fire_year_from_date)

# -------------------------------------------------------
# Filter 2020 fires in Colorado
# -------------------------------------------------------
fires = (
    mtbs_with_year
    .filterBounds(colorado)
    .filter(ee.Filter.eq("fire_year_from_date", FIRE_YEAR))
    .limit(N_FIRES)
)

print("Selected Colorado 2020 fires:", fires.size().getInfo())

# -------------------------------------------------------
# Add area and centroid metadata
# -------------------------------------------------------
def add_area(f):
    area_ha  = f.geometry().area(maxError=30).divide(10000)
    centroid = f.geometry().centroid(maxError=30).coordinates()
    return f.set({
        "event_id":    ee.String("fire_").cat(ee.String(f.id())),
        "fire_year":   FIRE_YEAR,
        "burn_area_ha": area_ha,
        "lon":         centroid.get(0),
        "lat":         centroid.get(1)
    })

fires = fires.map(add_area)

# -------------------------------------------------------
# AlphaEarth satellite embeddings (before / after / delta)
# -------------------------------------------------------
emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

emb_before_raw = emb.filterDate(f"{BEFORE_YEAR}-01-01", f"{BEFORE_YEAR}-12-31").first()
emb_after_raw  = emb.filterDate(f"{AFTER_YEAR}-01-01",  f"{AFTER_YEAR}-12-31").first()

# Reproject to safe CRS for extraction
emb_before = emb_before_raw.resample("bilinear").reproject(crs="EPSG:4326", scale=30)
emb_after  = emb_after_raw.resample("bilinear").reproject(crs="EPSG:4326", scale=30)

bands = emb_before.bandNames()

before = emb_before.rename(bands.map(lambda b: ee.String("before_").cat(ee.String(b))))
after  = emb_after.rename( bands.map(lambda b: ee.String("after_").cat(ee.String(b))))
delta  = emb_after.subtract(emb_before).rename(
             bands.map(lambda b: ee.String("delta_").cat(ee.String(b))))

embedding_stack = (
    before.addBands(after).addBands(delta)
    .reproject(crs="EPSG:4326", scale=30)
)

# -------------------------------------------------------
# GRIDMET fire-weather (7-day window around ignition)
# -------------------------------------------------------
gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")

def add_fire_weather(f):
    fire_date = ee.Date(f.get(FIRE_DATE_FIELD))
    window    = gridmet.filterDate(fire_date.advance(-3, "day"), fire_date.advance(4, "day"))

    vpd_day = gridmet.filterDate(fire_date, fire_date.advance(1, "day")).select("vpd").mean()
    vpd7    = window.select("vpd").mean()
    tmax7   = window.select("tmmx").mean()
    pr7     = window.select("pr").sum()

    def zonal_mean(img, band):
        return img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=f.geometry(),
            scale=4000,
            crs="EPSG:4326",
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4
        ).get(band)

    return f.set({
        "vpd_fire_day": zonal_mean(vpd_day, "vpd"),
        "vpd_7day":     zonal_mean(vpd7,    "vpd"),
        "tmmx_7day":    zonal_mean(tmax7,   "tmmx"),
        "pr_7day":      zonal_mean(pr7,     "pr")
    })

fires = fires.map(add_fire_weather)

# -------------------------------------------------------
# Extract embeddings over fire polygons and export
# -------------------------------------------------------
training = embedding_stack.reduceRegions(
    collection=fires,
    reducer=ee.Reducer.mean(),
    scale=30,
    crs="EPSG:4326",
    tileScale=8
)

geemap.ee_to_csv(training, OUT_CSV)
print(f"Saved: {OUT_CSV}")
```

### 3b. Centroid buffer extraction

Same pipeline but uses 1 km centroid buffers instead of full fire polygons,
which avoids geometry errors and produces cleaner zonal statistics.

```python
import ee
import geemap
import pandas as pd

ee.Initialize()

# Parameters
FIRE_YEAR   = 2020
BEFORE_YEAR = 2019
AFTER_YEAR  = 2021
N_FIRES     = 20
BUFFER_M    = 1000
OUT_CSV     = f"CO_{FIRE_YEAR}_MTBS_{N_FIRES}_AEF_GRIDMET_BUFFER_training_table.csv"

FIRE_DATE_FIELD = "Ig_Date"

states   = ee.FeatureCollection("TIGER/2018/States")
colorado = states.filter(ee.Filter.eq("NAME", "Colorado")).geometry()
mtbs     = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")

def add_fire_year_from_date(f):
    fire_date = ee.Date(f.get(FIRE_DATE_FIELD))
    return f.set({
        "fire_date":           fire_date.format("YYYY-MM-dd"),
        "fire_year_from_date": fire_date.get("year")
    })

mtbs_with_year = mtbs.map(add_fire_year_from_date)

fires_raw = (
    mtbs_with_year
    .filterBounds(colorado)
    .filter(ee.Filter.eq("fire_year_from_date", FIRE_YEAR))
    .limit(N_FIRES)
)

# -------------------------------------------------------
# Convert polygons to centroid buffers
# -------------------------------------------------------
def make_centroid_buffer(f):
    fire_date = ee.Date(f.get(FIRE_DATE_FIELD))
    lon = ee.Number.parse(f.get("BurnBndLon"))
    lat = ee.Number.parse(f.get("BurnBndLat"))

    point       = ee.Geometry.Point([lon, lat])
    buffer_geom = point.buffer(BUFFER_M)
    burn_area_ha = ee.Number(f.get("BurnBndAc")).multiply(0.404686)   # acres → ha

    return ee.Feature(buffer_geom).copyProperties(f).set({
        "event_id":    ee.String("fire_").cat(ee.String(f.id())),
        "fire_year":   FIRE_YEAR,
        "fire_date":   fire_date.format("YYYY-MM-dd"),
        "burn_area_ha": burn_area_ha,
        "lon":         lon,
        "lat":         lat,
        "buffer_m":    BUFFER_M
    })

fires = fires_raw.map(make_centroid_buffer)

# [Continue with same embedding + GRIDMET extraction as Section 3a]
# ... (embedding_stack, add_fire_weather, reduceRegions, ee_to_csv)
```

---

## 4. Full Training Dataset — Fire / Non-fire Labels

### 4a. Multi-year fire/non-fire with per-point climate and embeddings

Builds a balanced training dataset across 2000–2023. Each sample point
gets climate and embeddings computed dynamically from its own fire year.

```python
import ee
import pandas as pd
import os

ee.Initialize()

# ============================================================
# Settings
# ============================================================
SAFE_CRS         = "EPSG:4326"
SCALE            = 100
SEED             = 42

START_YEAR = 2000
END_YEAR   = 2023
N_FIRES    = 20

OUT_CSV = "xgboost_fire_nonfire_climate_topo_embedding.csv"

# ============================================================
# Study area
# ============================================================
states     = ee.FeatureCollection("TIGER/2018/States")
study_area = states.filter(ee.Filter.eq("NAME", "Colorado")).geometry()

# ============================================================
# Load and clean MTBS
# ============================================================
mtbs_all = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")
mtbs_bounds = mtbs_all.filterBounds(study_area)

print("MTBS fires in Colorado:", mtbs_bounds.size().getInfo())

def clean_fire(f):
    fire_year = ee.Date(f.get("Ig_Date")).get("year")
    geom = (
        f.geometry()
        .transform(SAFE_CRS, 1)
        .buffer(0, 1)
        .simplify(30)
    )
    return ee.Feature(geom, {
        "fire_id":   f.get("Event_ID"),
        "fire_name": f.get("Incid_Name"),
        "fire_year": fire_year
    })

fires_clean_all = mtbs_bounds.map(clean_fire)

fires_clean = (
    fires_clean_all
    .filter(ee.Filter.gte("fire_year", START_YEAR))
    .filter(ee.Filter.lte("fire_year", END_YEAR))
    .randomColumn("rand", SEED)
    .sort("rand")
    .limit(N_FIRES)
)

print("Selected fires:", fires_clean.size().getInfo())

# ============================================================
# Fire points (label = 1) — centroid of each polygon
# ============================================================
def make_fire_point(f):
    pt = f.geometry().centroid(30)
    return ee.Feature(pt, {
        "fire_id":     f.get("fire_id"),
        "fire_name":   f.get("fire_name"),
        "fire_year":   f.get("fire_year"),
        "label":       1,
        "sample_type": "fire"
    })

fire_points = fires_clean.map(make_fire_point)

# ============================================================
# Non-fire points (label = 0) — deterministic offset (~2 km)
# ============================================================
def make_nonfire_point(f):
    center = f.geometry().centroid(30)
    coords = center.coordinates()
    lon = ee.Number(coords.get(0))
    lat = ee.Number(coords.get(1))

    candidate_1 = ee.Geometry.Point([lon.add(0.02),      lat.add(0.02)])
    candidate_2 = ee.Geometry.Point([lon.subtract(0.02), lat.subtract(0.02)])

    nonfire_geom = ee.Algorithms.If(
        study_area.contains(candidate_1, 30),
        candidate_1,
        candidate_2
    )

    return ee.Feature(ee.Geometry(nonfire_geom), {
        "fire_id":     f.get("fire_id"),
        "fire_name":   f.get("fire_name"),
        "fire_year":   f.get("fire_year"),
        "label":       0,
        "sample_type": "non_fire"
    })

nonfire_points = fires_clean.map(make_nonfire_point)
samples        = fire_points.merge(nonfire_points)

print("Total sample points:", samples.size().getInfo())

# ============================================================
# Topography (static)
# ============================================================
dem    = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation")
slope  = ee.Terrain.slope(dem).rename("slope")
aspect = ee.Terrain.aspect(dem).rename("aspect")
topo_img = dem.addBands(slope).addBands(aspect)

# ============================================================
# TerraClimate — annual summary helper
# ============================================================
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")

def climate_for_year(year, prefix):
    year  = ee.Number(year)
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, "year")
    col   = terraclimate.filterDate(start, end)

    ppt  = col.select("pr").sum().rename(prefix + "_ppt_sum")
    tmax = col.select("tmmx").mean().multiply(0.1).rename(prefix + "_tmax_mean")
    tmin = col.select("tmmn").mean().multiply(0.1).rename(prefix + "_tmin_mean")
    vpd  = col.select("vpd").mean().multiply(0.01).rename(prefix + "_vpd_mean")

    return ppt.addBands(tmax).addBands(tmin).addBands(vpd)

# ============================================================
# AlphaEarth embeddings — annual mosaic helper
# ============================================================
embedding_ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

def rename_embedding(img, prefix):
    old_names = img.bandNames()
    new_names = old_names.map(
        lambda b: ee.String(prefix).cat("_").cat(ee.String(b))
    )
    return img.rename(new_names)

def embedding_for_year(year, prefix):
    year  = ee.Number(year)
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, "year")
    img   = (
        embedding_ic
        .filterDate(start, end)
        .filterBounds(study_area)
        .mosaic()
        .toFloat()
    )
    return rename_embedding(img, prefix)

# ============================================================
# Per-point sampling (fire_year-aware)
# ============================================================
def sample_one_point(f):
    fire_year  = ee.Number(f.get("fire_year"))
    pre_year   = fire_year.subtract(1)
    post_year  = fire_year.add(1)

    climate_pre  = climate_for_year(pre_year,  "pre")
    climate_post = climate_for_year(post_year, "post")

    emb_pre  = embedding_for_year(pre_year,  "pre_emb")
    emb_post = embedding_for_year(post_year, "post_emb")

    pre_raw  = (
        embedding_ic
        .filterDate(ee.Date.fromYMD(pre_year,  1, 1), ee.Date.fromYMD(pre_year.add(1),  1, 1))
        .filterBounds(study_area).mosaic().toFloat()
    )
    post_raw = (
        embedding_ic
        .filterDate(ee.Date.fromYMD(post_year, 1, 1), ee.Date.fromYMD(post_year.add(1), 1, 1))
        .filterBounds(study_area).mosaic().toFloat()
    )
    delta_emb = rename_embedding(post_raw.subtract(pre_raw), "delta_emb")

    predictor_img = (
        topo_img
        .addBands(climate_pre)
        .addBands(climate_post)
        .addBands(emb_pre)
        .addBands(emb_post)
        .addBands(delta_emb)
        .toFloat()
        .unmask(-9999)
        .setDefaultProjection(SAFE_CRS, None, SCALE)
    )

    return predictor_img.sampleRegions(
        collection=ee.FeatureCollection([f]),
        properties=["fire_id", "fire_name", "fire_year", "label", "sample_type"],
        scale=SCALE,
        projection=SAFE_CRS,
        geometries=True,
        tileScale=16
    ).first()

training = samples.map(sample_one_point)

# ============================================================
# Save to CSV
# ============================================================
training_dict = training.getInfo()
features      = training_dict.get("features", [])

rows = []
for feat in features:
    props = feat.get("properties", {}).copy()
    geom  = feat.get("geometry", None)
    if geom:
        props["longitude"] = geom["coordinates"][0]
        props["latitude"]  = geom["coordinates"][1]
    rows.append(props)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"Saved: {os.path.abspath(OUT_CSV)}")
print("Rows saved:", len(df))
if len(df) > 0:
    print("Label counts:\n", df["label"].value_counts())
```

### 4b. 2020 fires — nearby non-fire offset points

Simplified pipeline fixed to 2020. Non-fire points are placed ~2 km from
fire centroids using a coordinate offset.

```python
# Key settings
FIRE_YEAR           = 2020
PRE_YEAR            = 2019
POST_YEAR           = 2021
N_FIRES             = 50
NONFIRE_OFFSET_DEG  = 0.02   # ~2 km
OUT_CSV             = "xgboost_fire_nonfire_2020_50.csv"

# [Setup, MTBS filtering, fire/non-fire point creation, topography,
#  TerraClimate climate_for_year(), AlphaEarth raw_embedding_for_year()
#  and rename_embedding() are identical to Section 4a — fixed to 2020]

# -------------------------------------------------------
# Combine all predictors into a single image
# -------------------------------------------------------
predictor_img = (
    topo_img
    .addBands(climate_pre)
    .addBands(climate_post)
    .addBands(emb_pre)
    .addBands(emb_post)
    .addBands(emb_delta)
    .toFloat()
    .unmask(-9999)
    .setDefaultProjection(SAFE_CRS, None, SCALE)
)

# -------------------------------------------------------
# Sample all points at once (fixed-year pipeline)
# -------------------------------------------------------
training = predictor_img.sampleRegions(
    collection=samples,
    properties=["fire_id", "fire_name", "fire_year", "label", "sample_type"],
    scale=SCALE,
    projection=SAFE_CRS,
    geometries=True,
    tileScale=16
)
```

### 4c. 2020 fires — random non-fire points outside all MTBS burns

**Preferred approach.** Non-fire points are placed randomly anywhere in Colorado
that is outside all historical MTBS burn perimeters plus a 1 km safety buffer.
This prevents label contamination from previously burned areas.

```python
import ee
import pandas as pd
import os

ee.Initialize()

# ============================================================
# Settings
# ============================================================
SAFE_CRS            = "EPSG:4326"
SCALE               = 100
SEED                = 42

FIRE_YEAR           = 2020
PRE_YEAR            = 2019
POST_YEAR           = 2021

N_FIRES             = 50
N_NONFIRE_POINTS    = 200         # more non-fire for class balance
NONFIRE_MIN_DIST_M  = 1000        # 1 km exclusion around all MTBS burns

OUT_DIR = "mtbs_2020_prefire_20firepts_200nonfire"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV        = os.path.join(OUT_DIR, "fire_nonfire_2020_prefire_predictors.csv")
OUT_GPKG_FIRE  = os.path.join(OUT_DIR, "mtbs_2020_fire_perimeters.gpkg")

# ============================================================
# Study area and MTBS
# ============================================================
states     = ee.FeatureCollection("TIGER/2018/States")
study_area = states.filter(ee.Filter.eq("NAME", "Colorado")).geometry()

mtbs_all = (
    ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")
    .filterBounds(study_area)
)

print("All MTBS fires in Colorado:", mtbs_all.size().getInfo())

def clean_fire(f):
    fire_year = ee.Date(f.get("Ig_Date")).get("year")
    geom = (
        f.geometry()
        .transform(SAFE_CRS, 1)
        .buffer(0, 1)
        .simplify(60)
    )
    return ee.Feature(geom, {
        "fire_id":   f.get("Event_ID"),
        "fire_name": f.get("Incid_Name"),
        "fire_year": fire_year,
        "ig_date":   f.get("Ig_Date")
    })

mtbs_clean  = mtbs_all.map(clean_fire)
fires_2020  = (
    mtbs_clean
    .filter(ee.Filter.eq("fire_year", FIRE_YEAR))
    .randomColumn("rand", SEED)
    .sort("rand")
    .limit(N_FIRES)
)

print("Selected 2020 fires:", fires_2020.size().getInfo())

# ============================================================
# Fire points — random points within each 2020 burn polygon
# ============================================================
N_FIRE_POINTS_PER_FIRE = 20

def make_fire_points(f):
    pts = ee.FeatureCollection.randomPoints(
        region=f.geometry(),
        points=N_FIRE_POINTS_PER_FIRE,
        seed=SEED,
        maxError=30
    )
    return pts.map(
        lambda pt: ee.Feature(pt.geometry(), {
            "fire_id":     f.get("fire_id"),
            "fire_name":   f.get("fire_name"),
            "fire_year":   f.get("fire_year"),
            "label":       1,
            "sample_type": "fire"
        })
    )

fire_points = fires_2020.map(make_fire_points).flatten()
print("Fire points:", fire_points.size().getInfo())

# ============================================================
# Non-fire points — outside ALL historical MTBS + 1 km buffer
# ============================================================
all_mtbs_burn_geom  = mtbs_clean.geometry().buffer(0, 30)
burn_exclusion_geom = all_mtbs_burn_geom.buffer(NONFIRE_MIN_DIST_M, 30)

nonfire_area = (
    study_area
    .difference(burn_exclusion_geom, 30)
    .buffer(0, 30)
)

nonfire_raw = ee.FeatureCollection.randomPoints(
    region=nonfire_area,
    points=N_NONFIRE_POINTS,
    seed=SEED,
    maxError=30
)

nonfire_points = nonfire_raw.map(
    lambda f: ee.Feature(f.geometry(), {
        "fire_id":     "non_fire",
        "fire_name":   "non_fire",
        "fire_year":   FIRE_YEAR,
        "label":       0,
        "sample_type": "non_fire"
    })
)

# ============================================================
# Validate non-fire placement
# ============================================================
def check_nonfire(f):
    inside_any_burn = all_mtbs_burn_geom.contains(f.geometry(), 30)
    dist_to_burn    = f.geometry().distance(all_mtbs_burn_geom, 30)
    return f.set({
        "inside_any_mtbs_burn":              inside_any_burn,
        "distance_to_nearest_mtbs_burn_m":  dist_to_burn
    })

nonfire_points = nonfire_points.map(check_nonfire)

inside_count = nonfire_points.filter(
    ee.Filter.eq("inside_any_mtbs_burn", True)
).size().getInfo()

min_dist = nonfire_points.aggregate_min(
    "distance_to_nearest_mtbs_burn_m"
).getInfo()

print("Non-fire points inside any MTBS burn:", inside_count)
print("Minimum distance to nearest MTBS burn (m):", min_dist)

samples = fire_points.merge(nonfire_points)
print("Total sample points:", samples.size().getInfo())
```

---

## 5. Export Shapefiles

Export 2020 MTBS fire perimeters and validated non-fire points as
shapefiles for use in GIS.

```python
import ee
import geemap
import os

ee.Initialize()

SAFE_CRS           = "EPSG:4326"
FIRE_YEAR          = 2020
N_NONFIRE_POINTS   = 50
NONFIRE_MIN_DIST_M = 1000
SEED               = 42

OUT_DIR = "mtbs_2020_fire_nonfire_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# [Load study_area, mtbs_clean, mtbs_2020, nonfire_points_checked
#  using same logic as Section 4c]

# --- Export ---
geemap.ee_export_vector(
    mtbs_2020,
    filename=os.path.join(OUT_DIR, "mtbs_2020_fire_perimeters.shp")
)

geemap.ee_export_vector(
    nonfire_points_checked,
    filename=os.path.join(OUT_DIR, "nonfire_points_outside_any_year_mtbs.shp")
)
```

---

## 6. Final Training Dataset — Pre-fire predictors only

**This is the recommended dataset for modelling.** Uses only pre-fire
(2019) embeddings and climate to avoid data leakage. Outputs CSV, GeoPackage,
and shapefile.

```python
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import os

ee.Initialize()

# ============================================================
# Settings
# ============================================================
SAFE_CRS               = "EPSG:4326"
SCALE                  = 100
SEED                   = 42

FIRE_YEAR              = 2020
PRE_YEAR               = 2019

N_FIRES                = 50
N_FIRE_POINTS_PER_FIRE = 20
N_NONFIRE_POINTS       = 200
NONFIRE_MIN_DIST_M     = 1000

OUT_DIR = "mtbs_2020_prefire_20firepts_200nonfire"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV        = os.path.join(OUT_DIR, "fire_nonfire_2020_prefire_predictors.csv")
OUT_GPKG       = os.path.join(OUT_DIR, "fire_nonfire_2020_prefire_predictors.gpkg")
OUT_SHP        = os.path.join(OUT_DIR, "fire_nonfire_2020_prefire_predictors.shp")
OUT_GPKG_FIRE  = os.path.join(OUT_DIR, "mtbs_2020_fire_perimeters.gpkg")
OUT_SHP_FIRE   = os.path.join(OUT_DIR, "mtbs_2020_fire_perimeters.shp")

# ============================================================
# [Study area, MTBS loading, fire points, non-fire points]
# [Same as Section 4c — omitted for brevity]
# ============================================================

# ============================================================
# Topography
# ============================================================
dem    = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation")
slope  = ee.Terrain.slope(dem).rename("slope")
aspect = ee.Terrain.aspect(dem).rename("aspect")
topo_img = dem.addBands(slope).addBands(aspect)

# ============================================================
# Pre-burn TerraClimate (2019 only)
# ============================================================
terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")

def climate_for_year(year, prefix):
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, "year")
    col   = terraclimate.filterDate(start, end)

    ppt  = col.select("pr").sum().rename(prefix + "_ppt_sum")
    tmax = col.select("tmmx").mean().multiply(0.1).rename(prefix + "_tmax_mean")
    tmin = col.select("tmmn").mean().multiply(0.1).rename(prefix + "_tmin_mean")
    vpd  = col.select("vpd").mean().multiply(0.01).rename(prefix + "_vpd_mean")

    return ppt.addBands(tmax).addBands(tmin).addBands(vpd)

climate_pre = climate_for_year(PRE_YEAR, "pre")

# ============================================================
# Pre-burn AlphaEarth embedding (2019 only)
# ============================================================
embedding_ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

def raw_embedding_for_year(year):
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, "year")
    return (
        embedding_ic
        .filterDate(start, end)
        .filterBounds(study_area)
        .mosaic()
        .toFloat()
        .unmask(-9999)
    )

def rename_embedding(img, prefix):
    old_names = img.bandNames()
    new_names = old_names.map(
        lambda b: ee.String(prefix).cat("_").cat(ee.String(b))
    )
    return img.rename(new_names)

emb_pre = rename_embedding(raw_embedding_for_year(PRE_YEAR), "pre_emb")

# ============================================================
# Combine predictors
# ============================================================
predictor_img = (
    topo_img
    .addBands(climate_pre)
    .addBands(emb_pre)
    .toFloat()
    .unmask(-9999)
    .setDefaultProjection(SAFE_CRS, None, SCALE)
)

# ============================================================
# Sample predictors at all points
# ============================================================
training = predictor_img.sampleRegions(
    collection=samples,
    properties=[
        "fire_id", "fire_name", "fire_year",
        "label", "sample_type",
        "inside_any_mtbs_burn",
        "distance_to_nearest_mtbs_burn_m"
    ],
    scale=SCALE,
    projection=SAFE_CRS,
    geometries=True,
    tileScale=16
)

# ============================================================
# Helper: convert EE FeatureCollection to GeoDataFrame
# ============================================================
def ee_fc_to_gdf(fc):
    data = fc.getInfo()
    rows = []
    for feat in data["features"]:
        props = feat.get("properties", {}).copy()
        geom  = feat.get("geometry", None)
        if geom:
            props["geometry"] = shape(geom)
        rows.append(props)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")

# ============================================================
# Save point outputs
# ============================================================
gdf_points = ee_fc_to_gdf(training)
gdf_points["longitude"] = gdf_points.geometry.x
gdf_points["latitude"]  = gdf_points.geometry.y

pd.DataFrame(gdf_points.drop(columns="geometry")).to_csv(OUT_CSV, index=False)
gdf_points.to_file(OUT_GPKG, driver="GPKG")
gdf_points.to_file(OUT_SHP)

print("Rows saved:", len(gdf_points))
print("Label counts:\n", gdf_points["label"].value_counts())

embedding_cols = [c for c in gdf_points.columns if c.startswith("pre_emb_")]
print("Pre-embedding columns:", len(embedding_cols))

# ============================================================
# Save 2020 fire perimeters
# ============================================================
gdf_fire = ee_fc_to_gdf(fires_2020)
gdf_fire.to_file(OUT_GPKG_FIRE, driver="GPKG")
gdf_fire.to_file(OUT_SHP_FIRE)
```

---

## 7. XGBoost Classifier

Binary fire / non-fire classifier using **pre-fire only** predictors
(embeddings, climate, topography). 80/20 stratified train/test split.

```python
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv("xgboost_fire_nonfire_2020_50_clean_nonfire.csv")
df = df.replace(-9999, np.nan)

print("Rows:", len(df), "| Columns:", len(df.columns))
print("Label counts:\n", df["label"].value_counts())

# ============================================================
# 2. Define features — pre-fire only (no data leakage)
# ============================================================
pre_embedding_cols = [c for c in df.columns if c.startswith("pre_emb_")]

pre_climate_cols = [
    c for c in ["pre_ppt_sum", "pre_tmax_mean", "pre_tmin_mean", "pre_vpd_mean"]
    if c in df.columns
]

static_cols = [
    c for c in ["elevation", "slope", "aspect", "longitude", "latitude"]
    if c in df.columns
]

feature_cols = pre_embedding_cols + pre_climate_cols + static_cols

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["label"].astype(int)

print(f"Predictors: {len(pre_embedding_cols)} embedding | "
      f"{len(pre_climate_cols)} climate | {len(static_cols)} static")

# ============================================================
# 3. Train / test split (stratified 80/20)
# ============================================================
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# 4. Fit XGBoost
# ============================================================
clf = XGBClassifier(
    n_estimators=150,
    learning_rate=0.04,
    max_depth=2,
    min_child_weight=1,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42
)

clf.fit(X_train, y_train)

# ============================================================
# 5. Evaluate
# ============================================================
train_prob = clf.predict_proba(X_train)[:, 1]
test_prob  = clf.predict_proba(X_test)[:, 1]
train_pred = (train_prob >= 0.5).astype(int)
test_pred  = (test_prob  >= 0.5).astype(int)

print("\nModel diagnostics")
print("-" * 40)
print("Training accuracy:", round(accuracy_score(y_train, train_pred), 3))
print("Testing  accuracy:", round(accuracy_score(y_test,  test_pred),  3))
print("Training ROC AUC: ", round(roc_auc_score(y_train, train_prob),  3))
print("Testing  ROC AUC: ", round(roc_auc_score(y_test,  test_prob),   3))
print("Testing precision:", round(precision_score(y_test, test_pred),  3))
print("Testing recall:   ", round(recall_score(y_test,    test_pred),  3))
print("Testing F1:       ", round(f1_score(y_test,        test_pred),  3))
print("\n", classification_report(y_test, test_pred,
                                  target_names=["non_fire", "fire"]))

# ============================================================
# 6. Save predictions
# ============================================================
df_train = df_train.copy(); df_train["set"] = "train"
df_test  = df_test.copy();  df_test["set"]  = "test"

for split_df, prob, pred in [
    (df_train, train_prob, train_pred),
    (df_test,  test_prob,  test_pred)
]:
    split_df["pred_fire_probability"] = prob
    split_df["pred_label"]            = pred

df_pred = pd.concat([df_train, df_test])

# ============================================================
# 7. Feature importance
# ============================================================
importance = pd.DataFrame({
    "feature":    feature_cols,
    "importance": clf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 20 predictors:\n", importance.head(20))

# ============================================================
# 8. Visualizations
# ============================================================

# Confusion matrix
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Test confusion matrix")
plt.xticks([0, 1], ["Non-fire", "Fire"])
plt.yticks([0, 1], ["Non-fire", "Fire"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout(); plt.show()

# Predicted probability by class (boxplot)
plt.figure(figsize=(7, 5))
df_test.boxplot(column="pred_fire_probability", by="label")
plt.title("Test predicted fire probability by observed label")
plt.suptitle("")
plt.xlabel("Label: 0 = non-fire, 1 = fire")
plt.ylabel("Predicted fire probability")
plt.tight_layout(); plt.show()

# Histogram
plt.figure(figsize=(7, 5))
plt.hist(df_test.loc[df_test["label"] == 0, "pred_fire_probability"], alpha=0.6, label="Non-fire")
plt.hist(df_test.loc[df_test["label"] == 1, "pred_fire_probability"], alpha=0.6, label="Fire")
plt.xlabel("Predicted fire probability"); plt.ylabel("Count")
plt.title("Test predicted fire probability distribution")
plt.legend(); plt.tight_layout(); plt.show()

# Top 20 feature importance
top_imp = importance.head(20).sort_values("importance")
plt.figure(figsize=(8, 7))
plt.barh(top_imp["feature"], top_imp["importance"])
plt.xlabel("XGBoost feature importance")
plt.title("Top 20 predictors — pre-fire variables only")
plt.tight_layout(); plt.show()

# Spatial scatter — test set
plt.figure(figsize=(8, 7))
scatter = plt.scatter(
    df_test["longitude"], df_test["latitude"],
    c=df_test["pred_fire_probability"], s=70
)
for _, row in df_test.iterrows():
    plt.text(row["longitude"], row["latitude"], row["sample_type"], fontsize=8)
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Test points: predicted fire probability")
plt.colorbar(scatter, label="Predicted fire probability")
plt.tight_layout(); plt.show()

# ============================================================
# 9. Save outputs
# ============================================================
joblib.dump(clf,          "xgb_fire_probability_prefire_model.pkl")
joblib.dump(feature_cols, "feature_cols_prefire.pkl")
importance.to_csv("xgb_feature_importance_prefire.csv", index=False)
df_pred.to_csv("fire_nonfire_train_test_predictions_prefire.csv", index=False)

print("Saved: xgb_fire_probability_prefire_model.pkl")
```

---

## 8. Random Forest Classifier

Same pre-fire feature set and train/test split as Section 7, using a
Random Forest classifier instead of XGBoost.

```python
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, precision_score, recall_score, f1_score
)

# ============================================================
# 1–3. Load data and define features (same as Section 7)
# ============================================================
df = pd.read_csv("xgboost_fire_nonfire_2020_50_clean_nonfire.csv")
df = df.replace(-9999, np.nan)

pre_embedding_cols = [c for c in df.columns if c.startswith("pre_emb_")]
pre_climate_cols   = [c for c in ["pre_ppt_sum", "pre_tmax_mean",
                                   "pre_tmin_mean", "pre_vpd_mean"] if c in df.columns]
static_cols        = [c for c in ["elevation", "slope", "aspect",
                                   "longitude", "latitude"] if c in df.columns]
feature_cols       = pre_embedding_cols + pre_climate_cols + static_cols

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["label"].astype(int)

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.20, random_state=42, stratify=y
)

# ============================================================
# 4. Fit Random Forest
# ============================================================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ============================================================
# 5. Evaluate
# ============================================================
train_prob = rf.predict_proba(X_train)[:, 1]
test_prob  = rf.predict_proba(X_test)[:, 1]
train_pred = rf.predict(X_train)
test_pred  = rf.predict(X_test)

print("\nRandom Forest diagnostics")
print("-" * 40)
print("Training accuracy:", round(accuracy_score(y_train, train_pred), 3))
print("Testing  accuracy:", round(accuracy_score(y_test,  test_pred),  3))
print("Training ROC AUC: ", round(roc_auc_score(y_train, train_prob),  3))
print("Testing  ROC AUC: ", round(roc_auc_score(y_test,  test_prob),   3))
print("Testing F1:       ", round(f1_score(y_test, test_pred),         3))
print("\n", classification_report(y_test, test_pred,
                                  target_names=["non_fire", "fire"]))

# ============================================================
# 6–8. Feature importance and visualizations
# ============================================================
importance = pd.DataFrame({
    "feature":    feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

# [Confusion matrix, boxplot, histogram, spatial scatter —
#  identical structure to Section 7, labels updated to "Random Forest"]

# ============================================================
# 9. Save outputs
# ============================================================
joblib.dump(rf,           "rf_fire_probability_prefire_model.pkl")
joblib.dump(feature_cols, "rf_feature_cols_prefire.pkl")
importance.to_csv("rf_feature_importance_prefire.csv", index=False)

print("Saved: rf_fire_probability_prefire_model.pkl")
```

---

## 9. XGBoost Area Regression (LOO-CV)

Predicts **burned area (ha)** using leave-one-out cross-validation on the
20-fire training set. Area is log-transformed before fitting.

```python
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. Load data
# ============================================================
df = pd.read_csv("CO_2020_MTBS_20_AEF_GRIDMET_training_table.csv")
df = df.dropna(subset=["burn_area_ha"]).copy()
df["log_area"] = np.log1p(df["burn_area_ha"])

# ============================================================
# 2. Features: AEF embeddings + GRIDMET fire-weather
# ============================================================
feature_cols = [
    c for c in df.columns
    if c.startswith("before_") or c.startswith("after_") or c.startswith("delta_")
]

feature_cols += [
    c for c in ["vpd_fire_day", "vpd_7day", "tmmx_7day", "pr_7day"]
    if c in df.columns
]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["log_area"]

# ============================================================
# 3. XGBoost regressor with LOO-CV
# ============================================================
model = XGBRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    random_state=42
)

loo      = LeaveOneOut()
pred_log = cross_val_predict(model, X, y, cv=loo)
pred_area = np.expm1(pred_log)

print("LOO-CV performance")
print(f"MAE  : {mean_absolute_error(df['burn_area_ha'], pred_area):,.2f} ha")
print(f"RMSE : {mean_squared_error(df['burn_area_ha'], pred_area, squared=False):,.2f} ha")
print(f"R²   : {r2_score(df['burn_area_ha'], pred_area):.3f}")

# ============================================================
# 4. Fit final model on all data and save
# ============================================================
model.fit(X, y)

df["cv_pred_area_ha"] = pred_area

joblib.dump(model,        "xgb_fire_area_toy.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
df.to_csv("training_with_predictions.csv", index=False)

print("Saved: xgb_fire_area_toy.pkl")
```

---

## 10. Streamlit Digital Twin App

Interactive web app that lets users query AlphaEarth embeddings at any
Colorado location, adjust a VPD scenario slider, and get a real-time
burned area prediction from the trained XGBoost model.

```python
import ee
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ee.Initialize()

st.set_page_config(
    page_title="Embedding-informed Fire Digital Twin",
    layout="wide"
)
st.title("Embedding-informed fire digital twin demo")

# ============================================================
# Load model and training data
# ============================================================
model        = joblib.load("xgb_fire_area_toy.pkl")
feature_cols = joblib.load("feature_cols.pkl")
training_df  = pd.read_csv("training_with_predictions.csv")

emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

LATEST_EMBEDDING_YEAR = 2024
BASELINE_YEAR         = 2019

# ============================================================
# Helpers: fetch embedding from GEE
# ============================================================
def get_embedding_at_location(lon, lat, buffer_m, year, prefix):
    point  = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_m)
    img    = emb.filterDate(f"{year}-01-01", f"{year}-12-31").first()
    vals   = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    ).getInfo()
    return {f"{prefix}_{k}": v for k, v in vals.items()}


def build_new_location_row(lon, lat, buffer_m, user_vpd):
    before  = get_embedding_at_location(lon, lat, buffer_m, BASELINE_YEAR,         "before")
    current = get_embedding_at_location(lon, lat, buffer_m, LATEST_EMBEDDING_YEAR, "after")

    row = {}
    row.update(before)
    row.update(current)

    # Delta bands
    for bk in [k for k in row if k.startswith("before_")]:
        band = bk.replace("before_", "")
        ak = f"after_{band}"
        if ak in row:
            row[f"delta_{band}"] = row[ak] - row[bk]

    # Fire-weather scenario inputs
    row.update({
        "vpd_fire_day": user_vpd,
        "vpd_7day":     user_vpd,
        "tmmx_7day":    300,
        "pr_7day":      0
    })

    out = pd.DataFrame([row])
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0

    return out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


def predict_area(X):
    return np.expm1(model.predict(X)[0])

# ============================================================
# Sidebar controls
# ============================================================
mode = st.sidebar.radio(
    "Prediction mode",
    ["Existing demo fire", "New user location"]
)

user_vpd = st.sidebar.slider(
    "Current / scenario VPD (kPa)",
    min_value=0.0, max_value=8.0, value=2.5, step=0.1
)

# ============================================================
# Mode 1: existing demo fire from training set
# ============================================================
if mode == "Existing demo fire":
    fire_index = st.sidebar.selectbox("Select demo fire", training_df.index.tolist())
    row = training_df.loc[[fire_index]].copy()

    for col in ["vpd_7day", "vpd_fire_day"]:
        if col in row.columns:
            row[col] = user_vpd

    X         = row[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    pred_area = predict_area(X)
    observed  = row["burn_area_ha"].iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Observed burned area",            f"{observed:,.1f} ha")
    col2.metric("Scenario predicted area",         f"{pred_area:,.1f} ha")

    st.dataframe(row[["burn_area_ha", "vpd_7day", "vpd_fire_day"]])

# ============================================================
# Mode 2: new user-defined location
# ============================================================
else:
    lon      = st.sidebar.number_input("Longitude", value=-105.25, format="%.6f")
    lat      = st.sidebar.number_input("Latitude",  value=39.50,   format="%.6f")
    buffer_m = st.sidebar.slider(
        "Buffer radius around location (m)",
        min_value=250, max_value=5000, value=1000, step=250
    )

    if st.sidebar.button("Run new-location digital twin"):
        with st.spinner("Querying Earth Engine and running model..."):
            X_new     = build_new_location_row(lon, lat, buffer_m, user_vpd)
            pred_area = predict_area(X_new)

        st.subheader("New-location prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted affected area", f"{pred_area:,.1f} ha")
        c2.metric("Scenario VPD",            f"{user_vpd:.2f} kPa")
        c3.metric("Buffer radius",           f"{buffer_m:,} m")

        st.write(
            "Uses the most recent annual AlphaEarth embedding (2024) as the "
            "current-condition proxy, combined with the user-provided VPD scenario."
        )
        st.dataframe(X_new)
        st.download_button(
            "Download scenario inputs / outputs",
            X_new.assign(predicted_area_ha=pred_area).to_csv(index=False),
            "new_location_digital_twin_prediction.csv",
            "text/csv"
        )
    else:
        st.info("Enter lon/lat and VPD, then click Run.")
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Pre-fire predictors only | Prevents data leakage — the model should predict fire risk *before* the fire |
| Non-fire points outside all historical MTBS burns | Avoids labelling previously burned (recovered) areas as "safe" |
| 1 km safety buffer around MTBS perimeters | Removes ambiguous edge pixels near burn boundaries |
| AlphaEarth 64-D embedding | Captures vegetation type, canopy structure, fuel state from multi-sensor satellite data |
| TerraClimate annual summary | Long-term climate signal (drought, heat) as confounder control |
| Log-transform of burned area | Right-skewed fire size distribution; LOO-CV prevents overfitting on small N |
| Streamlit + GEE | Enables live Digital Twin queries at any location without pre-computing a raster |
