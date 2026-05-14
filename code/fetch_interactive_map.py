# fetch_interactive_map.py
# Prepare Google Earth Engine layers for the Colorado 2020 interactive map:
#   1. MTBS fire perimeters (vector)
#   2. MTBS fire severity raster (dNBR classes)
#   3. NLCD vegetation / land cover (most recent available)
#
# All operations stay server-side — no pixels are downloaded locally.
# Functions return styled ee objects ready to add to a geemap.Map().

import ee

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
EE_PROJECT  = "fire-344-467415"
STATE_NAME  = "Colorado"
FIRE_YEAR   = 2020

# MTBS severity class palette (0=unburned → 6=high severity)
SEVERITY_PALETTE = [
    "grey",     # 0 — outside/unburned
    "#006400",  # 1 — unburned / low
    "#7fffd4",  # 2 — low
    "#ffff00",  # 3 — moderate
    "#ff8c00",  # 4 — high
    "#ff0000",  # 5 — increased greenness
    "#8b0000",  # 6 — non-processing area mask
]

# NLCD 2021 land cover palette (21 classes)
NLCD_PALETTE = {
    11: "466b9f",   # Open water
    12: "d1def8",   # Perennial ice/snow
    21: "dec5c5",   # Developed, open space
    22: "d99282",   # Developed, low intensity
    23: "eb0000",   # Developed, medium intensity
    24: "ab0000",   # Developed, high intensity
    31: "b3ac9f",   # Barren rock
    41: "68ab5f",   # Deciduous forest
    42: "1c5f2c",   # Evergreen forest
    43: "b5c58f",   # Mixed forest
    51: "af963c",   # Dwarf scrub
    52: "ccb879",   # Shrub/scrub
    71: "dfdfc2",   # Grassland/herbaceous
    72: "d1d182",   # Sedge/herbaceous
    73: "a3cc51",   # Lichens
    74: "82ba9e",   # Moss
    81: "dcd939",   # Pasture/hay
    82: "ab6c28",   # Cultivated crops
    90: "b8d9eb",   # Woody wetlands
    95: "6c9fb8",   # Emergent herbaceous wetlands
}
NLCD_PALETTE_LIST = [v for v in NLCD_PALETTE.values()]


def init_ee(project: str = EE_PROJECT):
    ee.Initialize(project=project)


def get_colorado_boundary() -> ee.Geometry:
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.eq("NAME", STATE_NAME)).geometry()


# ----------------------------------------------------------
# Layer 1 — MTBS fire perimeters (vector)
# ----------------------------------------------------------

def get_fire_perimeters(boundary: ee.Geometry, fire_year: int = FIRE_YEAR) -> ee.FeatureCollection:
    """
    Return MTBS fire perimeters for Colorado filtered to fire_year.
    Adds fire_year derived from Ig_Date and burn area in hectares.
    """
    mtbs = ee.FeatureCollection("USFS/GTAC/MTBS/burned_area_boundaries/v1")

    def enrich(f):
        year    = ee.Date(f.get("Ig_Date")).get("year")
        area_ha = f.geometry().area(maxError=30).divide(10000)
        return f.set({
            "fire_year": year,
            "area_ha":   area_ha,
            "fire_name": f.get("Incid_Name"),
        })

    return (
        mtbs
        .filterBounds(boundary)
        .map(enrich)
        .filter(ee.Filter.eq("fire_year", fire_year))
    )


def style_perimeters(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    """Apply a visible outline style to fire perimeters."""
    return fc.style(
        color="FF4500",
        fillColor="FF450033",
        width=2,
    )


# ----------------------------------------------------------
# Layer 2 — MTBS fire severity raster
# ----------------------------------------------------------

def get_severity_image(boundary: ee.Geometry, fire_year: int = FIRE_YEAR) -> ee.Image:
    """
    Return the MTBS annual burn severity mosaic clipped to Colorado for fire_year.
    Band: 'burnSeverity' — integer class values 0–6.
    """
    severity_col = ee.ImageCollection("USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1")

    img = (
        severity_col
        .filter(ee.Filter.calendarRange(fire_year, fire_year, "year"))
        .first()
        .select("Severity")
        .clip(boundary)
    )
    return img


def severity_vis_params() -> dict:
    return {
        "min": 0,
        "max": 6,
        "palette": SEVERITY_PALETTE,
    }


# ----------------------------------------------------------
# Layer 3 — NLCD vegetation / land cover
# ----------------------------------------------------------

def get_nlcd_image(boundary: ee.Geometry, year: int = 2021) -> ee.Image:
    """
    Return the NLCD land cover image clipped to Colorado.
    Tries the requested year; falls back to 2019 if not available.
    Available NLCD years in EE: 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021.
    Band: 'landcover'
    """
    nlcd = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")

    img = (
        nlcd
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .first()
        .select("landcover")
        .clip(boundary)
    )
    return img


def nlcd_vis_params() -> dict:
    return {
        "min": 11,
        "max": 95,
        "palette": NLCD_PALETTE_LIST,
    }


# ----------------------------------------------------------
# Convenience: return all three layers at once
# ----------------------------------------------------------

def load_all_layers(fire_year: int = FIRE_YEAR):
    """
    Initialize EE and return all three map layers + the Colorado boundary.

    Returns
    -------
    dict with keys:
        boundary    : ee.Geometry
        perimeters  : ee.FeatureCollection  (raw, for popups)
        perimeters_styled : ee.FeatureCollection (styled image)
        severity    : ee.Image
        nlcd        : ee.Image
        severity_vis: dict
        nlcd_vis    : dict
    """
    colorado   = get_colorado_boundary()
    perimeters = get_fire_perimeters(colorado, fire_year)
    severity   = get_severity_image(colorado, fire_year)
    nlcd       = get_nlcd_image(colorado)

    return {
        "boundary":          colorado,
        "perimeters":        perimeters,
        "perimeters_styled": style_perimeters(perimeters),
        "severity":          severity,
        "nlcd":              nlcd,
        "severity_vis":      severity_vis_params(),
        "nlcd_vis":          nlcd_vis_params(),
    }


# ----------------------------------------------------------
# NLCD class lookup (for legend / popups)
# ----------------------------------------------------------

NLCD_CLASSES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Rock/Sand/Clay",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    51: "Dwarf Scrub",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    72: "Sedge/Herbaceous",
    73: "Lichens",
    74: "Moss",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands",
}

SEVERITY_CLASSES = {
    0: "Outside/No Data",
    1: "Unburned to Low",
    2: "Low Severity",
    3: "Moderate Severity",
    4: "High Severity",
    5: "Increased Greenness",
    6: "Non-processing Area Mask",
}
