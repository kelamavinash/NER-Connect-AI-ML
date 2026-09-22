"""Independent hazard task contracts; no opaque combined safety score."""

LANDSLIDE_FEATURES = (
    "slope_deg",
    "elevation_m",
    "aspect_deg",
    "curvature",
    "terrain_ruggedness",
    "soil_type",
    "geology",
    "land_cover",
    "distance_to_river_m",
    "distance_to_drainage_m",
    "historical_landslide_density",
    "rain_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_24h_mm",
    "rain_72h_mm",
    "forecast_rain_6h_mm",
    "soil_moisture",
)

FLOOD_FEATURES = (
    "rain_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "rain_24h_mm",
    "rain_72h_mm",
    "forecast_rain_6h_mm",
    "elevation_m",
    "slope_deg",
    "distance_to_river_m",
    "distance_to_drainage_m",
    "flow_accumulation",
    "drainage_density",
    "floodplain_exposure",
    "soil_permeability",
    "historical_flood_exposure",
    "land_cover",
    "soil_type",
)

HAZARD_TASKS = {
    "landslide": {
        "features": LANDSLIDE_FEATURES,
        "target": "documented landslide occurrence linked in space/time to the road segment",
        "status": "labels_required",
    },
    "flood": {
        "features": FLOOD_FEATURES,
        "target": "documented road accessibility impact or inundation",
        "status": "labels_required_heuristic_live",
    },
}
