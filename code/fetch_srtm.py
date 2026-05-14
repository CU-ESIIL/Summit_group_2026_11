# fetch_srtm.py
# Extract SRTM digital elevation data (elevation, slope, aspect) for Colorado
# via Google Earth Engine. Samples values at provided point locations or
# exports a summary CSV.

import ee
import pandas as pd
import os

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
EE_PROJECT  = "fire-344-467415"
STATE_NAME  = "Colorado"
SCALE       = 90        # meters — SRTM native resolution is ~30m; use 90 for speed
SAFE_CRS    = "EPSG:4326"
OUT_CSV     = "srtm_colorado_sample.csv"


def init_ee(project: str = EE_PROJECT):
    ee.Initialize(project=project)


def get_colorado_boundary() -> ee.Geometry:
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.eq("NAME", STATE_NAME)).geometry()


def build_topo_image() -> ee.Image:
    """Return a 3-band image: elevation, slope, aspect."""
    dem    = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation")
    slope  = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    return dem.addBands(slope).addBands(aspect)


def sample_topo_at_points(
    points: ee.FeatureCollection,
    topo: ee.Image,
    scale: int = SCALE,
) -> pd.DataFrame:
    """
    Sample elevation/slope/aspect at a FeatureCollection of points.
    The input FeatureCollection should have a 'fire_id' and 'fire_year' property.
    """
    sampled = topo.sampleRegions(
        collection=points,
        properties=["fire_id", "fire_year", "label"],
        scale=scale,
        projection=SAFE_CRS,
        geometries=True,
    )

    info = sampled.getInfo()
    rows = []
    for feat in info["features"]:
        props = feat["properties"].copy()
        geom  = feat.get("geometry", {})
        if geom:
            coords = geom["coordinates"]
            props["longitude"] = coords[0]
            props["latitude"]  = coords[1]
        rows.append(props)

    return pd.DataFrame(rows)


def summarize_topo_for_colorado(topo: ee.Image, boundary: ee.Geometry) -> dict:
    """
    Compute mean elevation, slope, and aspect over all of Colorado.
    Returns a plain dict of band -> mean value.
    """
    stats = topo.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=boundary,
        scale=1000,         # coarse pass for a state-level summary
        maxPixels=1e9,
        bestEffort=True,
    )
    return stats.getInfo()


def main():
    print("Initializing Earth Engine...")
    init_ee()

    print("Loading Colorado boundary...")
    colorado = get_colorado_boundary()

    print("Building topography image (elevation, slope, aspect)...")
    topo = build_topo_image()

    print("Computing state-level topo summary...")
    summary = summarize_topo_for_colorado(topo, colorado)
    print("Colorado topo summary (mean values):")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")

    # --- Optional: sample at MTBS fire centroids ---
    # Uncomment and pass in a FeatureCollection of points if needed.
    #
    # from fetch_mtbs import init_ee, load_mtbs, get_colorado_boundary
    # fires = load_mtbs(colorado, 2000, 2023)
    # def make_centroid(f):
    #     return ee.Feature(f.geometry().centroid(30), {
    #         "fire_id":   f.get("Event_ID"),
    #         "fire_year": f.get("fire_year"),
    #         "label":     1
    #     })
    # centroids = fires.map(make_centroid)
    # df = sample_topo_at_points(centroids, topo)
    # df.to_csv(OUT_CSV, index=False)
    # print(f"CSV saved: {os.path.abspath(OUT_CSV)}")
    # print(df.head())

    return summary


if __name__ == "__main__":
    main()
