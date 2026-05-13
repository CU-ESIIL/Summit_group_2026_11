# configuration.py
# Central configuration for Summit Group 2026-11 analysis scripts.
# Edit values here to adjust defaults across prism_quicklook.py and single_hull_demo.py.

# --- PRISM data settings ---
PRISM_DEFAULT_VARIABLE = "tmax"       # e.g. "tmax", "tmin", "ppt", "tmean"
PRISM_DEFAULT_RESOLUTION = "800m"    # "4km", "800m", or "400m"
PRISM_DEFAULT_FREQ = "daily"         # "daily", "monthly", or "annual"
PRISM_DEFAULT_REGION = "us"
PRISM_DEFAULT_NETWORK = "an"

# Default spatial bounding box [minx, miny, maxx, maxy] (WGS84)
PRISM_DEFAULT_BBOX = [-106.0, 39.0, -104.5, 40.5]  # Colorado Front Range

# --- Hull visualization settings ---
HULL_N_RING_SAMPLES = 200     # Points sampled along each daily fire ring
HULL_N_THETA = 128            # Angular resolution of the ruled surface
HULL_CRS_EPSG = 5070          # Projected CRS for distance calculations (CONUS Albers)
HULL_SMOOTH_OVER_Z = 3        # Smoothing window across event days (odd int or 1 to disable)
HULL_CMAP = "cividis"         # Colormap for hull surface
HULL_WALL_ALPHA = 0.35        # Transparency of hull faces
HULL_EDGE_ALPHA = 0.25        # Transparency of hull edges
HULL_ELEV = 26                # 3D view elevation angle
HULL_AZIM = -58               # 3D view azimuth angle
HULL_FIGSIZE = (9, 8)         # Figure size in inches

# --- Column name defaults ---
DATE_COL = "date"
Z_COL = "event_day"
