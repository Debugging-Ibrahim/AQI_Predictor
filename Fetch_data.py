# -*- coding: utf-8 -*-
"""
Unified AQI data pipeline for Faisalabad.

Two entry points, one shared engine:
  - run_backfill()   -> historical range, WITH targets, for training
  - run_live_fetch()  -> recent hours only, NO targets, for the feature store

Both use the same robust fetch client (caching + retry) and the same
engineer_features() function, so features are computed identically
whether it's 2025 historical data or this hour's live reading.
"""

import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

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

# Robust client -- shared by both modes (caching avoids re-hitting the API
# for identical requests, retry handles flaky connections)
cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# ---------- Shared fetch + parse ----------

def response_to_dataframe(response, variable_names):
    hourly = response.Hourly()
    start = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    freq = pd.Timedelta(seconds=hourly.Interval())
    dates = pd.date_range(start=start, end=end, freq=freq, inclusive="left")

    data = {"timestamp": dates}
    for i, name in enumerate(variable_names):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def fetch_weather_and_aq(weather_url, weather_params, aq_params):
    """One call each, merged on timestamp. Works for archive OR forecast URLs."""
    weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
    aq_response = openmeteo.weather_api(AQ_URL, params=aq_params)[0]

    df_weather = response_to_dataframe(weather_response, WEATHER_FIELDS)
    df_aq = response_to_dataframe(aq_response, AIR_QUALITY_FIELDS)

    # inner join: we only want hours where BOTH weather and AQ are present
    df = pd.merge(df_weather, df_aq, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["city"] = CITY_NAME
    return df


# ---------- Feature engineering (identical for both modes) ----------

def engineer_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    hour = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["us_aqi_lag_6h"] = df["us_aqi"].shift(6)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["us_aqi_lag_48h"] = df["us_aqi"].shift(48)
    df["us_aqi_lag_72h"] = df["us_aqi"].shift(72)
    df["us_aqi_rolling_mean_24h"] = df["us_aqi"].rolling(window=24).mean()
    df["us_aqi_rolling_mean_72h"] = df["us_aqi"].rolling(window=72).mean()
    df["aqi_change_rate_1h"] = df["us_aqi"] - df["us_aqi_lag_1h"]

    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["pm2_5_lag_48h"] = df["pm2_5"].shift(48)
    df["pm2_5_lag_72h"] = df["pm2_5"].shift(72)
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].rolling(window=24).mean()

    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_24h"] = df["pm10"].shift(24)

    df["temperature_2m_lag_1h"] = df["temperature_2m"].shift(1)
    df["temperature_2m_lag_24h"] = df["temperature_2m"].shift(24)
    df["wind_speed_10m_lag_1h"] = df["wind_speed_10m"].shift(1)
    df["wind_speed_10m_lag_6h"] = df["wind_speed_10m"].shift(6)
    df["relative_humidity_2m_lag_1h"] = df["relative_humidity_2m"].shift(1)
    df["surface_pressure_lag_1h"] = df["surface_pressure"].shift(1)

    df["precipitation_rolling_sum_6h"] = df["precipitation"].rolling(window=6).sum()
    df["precipitation_rolling_sum_24h"] = df["precipitation"].rolling(window=24).sum()
    df["rain_rolling_sum_6h"] = df["rain"].rolling(window=6).sum()
    df["rain_rolling_sum_24h"] = df["rain"].rolling(window=24).sum()

    return df


def add_targets(df):
    """Only ever call this on historical data, where 'the future' already happened."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_day1"] = df["us_aqi"].shift(-24)
    df["aqi_day2"] = df["us_aqi"].shift(-48)
    df["aqi_day3"] = df["us_aqi"].shift(-72)
    return df


# ---------- Mode 1: Backfill (for training) ----------

def run_backfill(start_date="2025-01-01", end_date="2026-08-18"):
    weather_params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_FIELDS), "timezone": "UTC",
    }
    aq_params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(AIR_QUALITY_FIELDS), "timezone": "UTC",
    }
    df = fetch_weather_and_aq(ARCHIVE_URL, weather_params, aq_params)
    df = engineer_features(df)
    df = add_targets(df)               # safe here -- future already happened

    feature_cols = [c for c in df.columns
                     if c not in ["timestamp", "city", "aqi_day1", "aqi_day2", "aqi_day3"]]
    target_cols = ["aqi_day1", "aqi_day2", "aqi_day3"]
    df_clean = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    print(f"Backfill: {len(df)} raw hours -> {len(df_clean)} training-ready rows")
    return df_clean


# ---------- Mode 2: Live fetch (for the feature store, every 3h) ----------

def run_live_fetch(past_days=4):
    """past_days must be >= 3 so 72h lags can be computed for the newest row."""
    weather_params = {
        "latitude": LAT, "longitude": LON, "past_days": past_days,
        "forecast_days": 1,   # today only -- never pulls future forecast
        "hourly": ",".join(WEATHER_FIELDS), "timezone": "UTC",
    }
    aq_params = {
        "latitude": LAT, "longitude": LON, "past_days": past_days,
        "forecast_days": 1,
        "hourly": ",".join(AIR_QUALITY_FIELDS), "timezone": "UTC",
    }
    df = fetch_weather_and_aq(FORECAST_URL, weather_params, aq_params)
    df = engineer_features(df)         # NO add_targets() here -- future doesn't exist yet

    # Only the newest row(s) are new; older ones already sit in Hopsworks
    latest_row = df.tail(1)
    print(f"Live fetch: {len(df)} hours pulled, pushing {len(latest_row)} new row(s)")
    return latest_row


if __name__ == "__main__":
    # For your first backfill run:
    training_df = run_backfill()
    print(training_df.tail())

    # For your recurring 3-hourly GitHub Action:
    # new_row = run_live_fetch(past_days=4)