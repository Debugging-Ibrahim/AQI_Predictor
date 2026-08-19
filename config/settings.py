# -*- coding: utf-8 -*-
"""
Central configuration settings for AQI Predictor.
"""

LAT, LON = 31.4187, 73.0791
CITY_NAME = "Faisalabad"

WEATHER_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]

AIR_QUALITY_FIELDS = [
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "sulphur_dioxide", "carbon_monoxide", "dust",
    "aerosol_optical_depth", "us_aqi",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

SUPABASE_TABLE_NAME = "aqi_features"
