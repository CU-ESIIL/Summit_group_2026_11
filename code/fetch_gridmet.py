# fetch_gridmet.py
# Extract GridMET climate data for Colorado via Google Earth Engine.
# Variables: tmax, tmin, precipitation, VPD, wind speed.
# Outputs a CSV of spatial means over Colorado by date.

import ee
import pandas as pd
import os
from typing import List, Optional

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
EE_PROJECT  = "fire-344-467415"
STATE_NAME  = "Colorado"
OUT_CSV     = "gridmet_colorado_climate.csv"

# GridMET band names and human-readable labels
GRIDMET_BANDS = {
    "tmmx": "tmax_C",       # max temperature (K -> C)
    "tmmn": "tmin_C",       # min temperature (K -> C)
    "pr":   "precip_mm",    # precipitation (mm)
    "vpd":  "vpd_kPa",      # vapor pressure deficit (kPa)
    "vs":   "wind_ms",      # wind speed (m/s)
    "erc":  "erc",          # energy release component (fire weather)
}


# ----------------------------------------------------------
# Init
# ----------------------------------------------------------

def init_ee(project: str = EE_PROJECT):
    ee.Initialize(project=project)


def get_colorado_boundary() -> ee.Geometry:
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.eq("NAME", STATE_NAME)).geometry()


# ----------------------------------------------------------
# Daily spatial mean over Colorado
# ----------------------------------------------------------

def extract_daily_means(
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
    boundary: Optional[ee.Geometry] = None,
) -> pd.DataFrame:
    """
    Compute daily spatial mean of GridMET variables over Colorado.

    Parameters
    ----------
    start_date : str  e.g. "2020-01-01"
    end_date   : str  e.g. "2020-12-31"
    bands      : list of GridMET band names (defaults to all in GRIDMET_BANDS)
    boundary   : ee.Geometry (defaults to Colorado)

    Returns
    -------
    pd.DataFrame with columns [date, tmax_C, tmin_C, precip_mm, vpd_kPa, ...]
    """
    if bands is None:
        bands = list(GRIDMET_BANDS.keys())
    if boundary is None:
        boundary = get_colorado_boundary()

    gridmet = (
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterDate(start_date, end_date)
        .select(bands)
    )

    def image_mean(img):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=boundary,
            scale=4000,
            maxPixels=1e9,
            bestEffort=True,
        )
        return ee.Feature(None, stats).set("date", img.date().format("YYYY-MM-dd"))

    fc    = gridmet.map(image_mean)
    info  = fc.getInfo()
    rows  = [f["properties"] for f in info["features"]]
    df    = pd.DataFrame(rows)

    # Convert temperatures from Kelvin to Celsius
    if "tmmx" in df.columns:
        df["tmax_C"] = df["tmmx"] - 273.15
        df.drop(columns=["tmmx"], inplace=True)
    if "tmmn" in df.columns:
        df["tmin_C"] = df["tmmn"] - 273.15
        df.drop(columns=["tmmn"], inplace=True)

    # Rename remaining bands
    rename_map = {k: v for k, v in GRIDMET_BANDS.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ----------------------------------------------------------
# Fire-weather window: extract mean over N days around fire date
# ----------------------------------------------------------

def extract_fire_weather(
    fire_date: str,
    lon: float,
    lat: float,
    buffer_m: int = 5000,
    window_days: int = 7,
    bands: Optional[List[str]] = None,
) -> dict:
    """
    Extract mean GridMET climate variables within a buffer around a fire
    point over a window centered on the fire date.

    Parameters
    ----------
    fire_date  : str   e.g. "2020-09-15"
    lon, lat   : float  fire centroid coordinates
    buffer_m   : int    radius in meters around the point
    window_days: int    days before the fire date to average over
    bands      : list of GridMET band names

    Returns
    -------
    dict of band -> mean value
    """
    if bands is None:
        bands = list(GRIDMET_BANDS.keys())

    fire_dt  = ee.Date(fire_date)
    start    = fire_dt.advance(-window_days, "day")
    end      = fire_dt.advance(1, "day")
    region   = ee.Geometry.Point([lon, lat]).buffer(buffer_m)

    gridmet  = (
        ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
        .filterDate(start, end)
        .select(bands)
    )

    mean_img = gridmet.mean()

    stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=4000,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    # Convert temperatures
    if "tmmx" in stats and stats["tmmx"] is not None:
        stats["tmax_C"] = stats.pop("tmmx") - 273.15
    if "tmmn" in stats and stats["tmmn"] is not None:
        stats["tmin_C"] = stats.pop("tmmn") - 273.15

    return stats


# ----------------------------------------------------------
# Monthly aggregation helper
# ----------------------------------------------------------

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a daily DataFrame to monthly means/sums."""
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    agg = {}
    for col in df.columns:
        if col in ("date", "year_month"):
            continue
        # precipitation is summed; everything else is averaged
        agg[col] = "sum" if col == "precip_mm" else "mean"

    monthly = df.groupby("year_month").agg(agg).reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    return monthly


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():
    print("Initializing Earth Engine...")
    init_ee()

    print("Loading Colorado boundary...")
    colorado = get_colorado_boundary()

    print("Extracting daily GridMET means for Colorado (2019–2021)...")
    df = extract_daily_means(
        start_date="2019-01-01",
        end_date="2021-12-31",
        boundary=colorado,
    )

    df.to_csv(OUT_CSV, index=False)
    print(f"\nDaily CSV saved: {os.path.abspath(OUT_CSV)}")
    print(f"Shape: {df.shape}")
    print(df.head())

    # Monthly summary
    monthly = aggregate_monthly(df)
    monthly_csv = OUT_CSV.replace(".csv", "_monthly.csv")
    monthly.to_csv(monthly_csv, index=False)
    print(f"\nMonthly CSV saved: {os.path.abspath(monthly_csv)}")
    print(monthly.head(12))

    # Example: fire-weather window for a single fire
    print("\nExample fire-weather extraction (East Troublesome Fire, 2020-10-14):")
    weather = extract_fire_weather(
        fire_date="2020-10-14",
        lon=-105.88,
        lat=40.27,
        buffer_m=5000,
        window_days=7,
    )
    for k, v in weather.items():
        print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: None")

    return df


if __name__ == "__main__":
    main()
