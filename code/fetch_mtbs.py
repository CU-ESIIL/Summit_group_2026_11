# fetch_mtbs.py
# Extract MTBS fire perimeter data for Colorado via Google Earth Engine.
# Outputs a CSV of fire attributes and a GeoJSON of fire polygons.

import ee
import geemap
import pandas as pd
import json
import os

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
EE_PROJECT    = "fire-344-467415"
STATE_NAME    = "Colorado"
START_YEAR    = 2000
END_YEAR      = 2000
OUT_CSV       = "mtbs_colorado.csv"
OUT_GEOJSON   = "mtbs_colorado.geojson"


def init_ee(project: str = EE_PROJECT):
    ee.Initialize(project=project)


def get_colorado_boundary() -> ee.Geometry:
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.eq("NAME", STATE_NAME)).geometry()


def load_mtbs(boundary: ee.Geometry, start_year: int, end_year: int) -> ee.FeatureCollection:
    """Filter MTBS to Colorado and the given year range."""
    mtbs = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")

    def add_fire_year(f):
        return f.set("fire_year", ee.Date(f.get("Ig_Date")).get("year"))

    return (
        mtbs
        .filterBounds(boundary)
        .map(add_fire_year)
        .filter(ee.Filter.gte("fire_year", start_year))
        .filter(ee.Filter.lte("fire_year", end_year))
    )


def mtbs_to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
    """Convert EE FeatureCollection to a pandas DataFrame (attributes only)."""
    info = fc.getInfo()
    rows = []
    for feat in info["features"]:
        props = feat["properties"].copy()
        geom  = feat.get("geometry", {})
        if geom:
            coords = geom.get("coordinates", [])
            props["geometry_type"] = geom.get("type", "")
        rows.append(props)
    return pd.DataFrame(rows)


def save_geojson(fc: ee.FeatureCollection, path: str):
    """Save FeatureCollection as GeoJSON."""
    info = fc.getInfo()
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"GeoJSON saved: {os.path.abspath(path)}")


def main():
    print("Initializing Earth Engine...")
    init_ee()

    print("Loading Colorado boundary...")
    colorado = get_colorado_boundary()

    print(f"Loading MTBS fires {START_YEAR}–{END_YEAR}...")
    fires = load_mtbs(colorado, START_YEAR, END_YEAR)
    print(f"Total fires found: {fires.size().getInfo()}")

    print("Converting to DataFrame...")
    df = mtbs_to_dataframe(fires)
    df.to_csv(OUT_CSV, index=False)
    print(f"CSV saved:     {os.path.abspath(OUT_CSV)}")
    print(f"Columns:       {df.columns.tolist()}")
    print(f"Shape:         {df.shape}")

    print("Saving GeoJSON...")
    save_geojson(fires, OUT_GEOJSON)

    return df


if __name__ == "__main__":
    df = main()
    print(df.head())
