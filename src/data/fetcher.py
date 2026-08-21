# -*- coding: utf-8 -*-
"""
Open-Meteo API Fetcher and Feature Engineering Engine.
"""

import numpy as np
import pandas as pd
import requests
import openmeteo_requests
from retry_requests import retry

from config.settings import (
    LAT, LON, CITY_NAME,
    WEATHER_FIELDS, AIR_QUALITY_FIELDS,
    ARCHIVE_URL, FORECAST_URL, AQ_URL
)

# Robust client setup
plain_session = requests.Session()
retry_session = retry(plain_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


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
    """Fetches weather and air quality datasets and merges them on timestamp."""
    weather_response = openmeteo.weather_api(weather_url, params=weather_params)[0]
    aq_response = openmeteo.weather_api(AQ_URL, params=aq_params)[0]

    df_weather = response_to_dataframe(weather_response, WEATHER_FIELDS)
    df_aq = response_to_dataframe(aq_response, AIR_QUALITY_FIELDS)

    df = pd.merge(df_weather, df_aq, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["city"] = CITY_NAME
    return df


def engineer_features(df):
    """Computes time-cyclic features, lags, rolling statistics, and change rates."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    hour = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month

    # Time Cyclic Encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # US AQI Lags & Rolling Means
    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["us_aqi_lag_6h"] = df["us_aqi"].shift(6)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["us_aqi_lag_48h"] = df["us_aqi"].shift(48)
    df["us_aqi_lag_72h"] = df["us_aqi"].shift(72)
    df["us_aqi_rolling_mean_24h"] = df["us_aqi"].rolling(window=24).mean()
    df["us_aqi_rolling_mean_72h"] = df["us_aqi"].rolling(window=72).mean()
    df["aqi_change_rate_1h"] = df["us_aqi"] - df["us_aqi_lag_1h"]

    # PM2.5 Lags & Rolling Means
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["pm2_5_lag_48h"] = df["pm2_5"].shift(48)
    df["pm2_5_lag_72h"] = df["pm2_5"].shift(72)
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].rolling(window=24).mean()

    # PM10 Lags
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_24h"] = df["pm10"].shift(24)

    # Weather Contextual Lags
    df["temperature_2m_lag_1h"] = df["temperature_2m"].shift(1)
    df["temperature_2m_lag_24h"] = df["temperature_2m"].shift(24)
    df["wind_speed_10m_lag_1h"] = df["wind_speed_10m"].shift(1)
    df["wind_speed_10m_lag_6h"] = df["wind_speed_10m"].shift(6)
    df["relative_humidity_2m_lag_1h"] = df["relative_humidity_2m"].shift(1)
    df["surface_pressure_lag_1h"] = df["surface_pressure"].shift(1)

    # Precipitation & Rain Cumulative Sums
    df["precipitation_rolling_sum_6h"] = df["precipitation"].rolling(window=6).sum()
    df["precipitation_rolling_sum_24h"] = df["precipitation"].rolling(window=24).sum()
    df["rain_rolling_sum_6h"] = df["rain"].rolling(window=6).sum()
    df["rain_rolling_sum_24h"] = df["rain"].rolling(window=24).sum()

    return df


def add_targets(df):
    """Computes target variables for 1-day, 2-day, and 3-day future AQI forecasts."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_day1"] = df["us_aqi"].shift(-24)
    df["aqi_day2"] = df["us_aqi"].shift(-48)
    df["aqi_day3"] = df["us_aqi"].shift(-72)
    return df


def run_backfill(start_date="2025-01-01", end_date=None):
    """Fetches historical range, engineers features, and creates target columns."""
    if end_date is None:
        end_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

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
    df = add_targets(df)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "city", "aqi_day1", "aqi_day2", "aqi_day3"]]
    target_cols = ["aqi_day1", "aqi_day2", "aqi_day3"]
    df_clean = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    print(f"Backfill: {len(df)} raw hours -> {len(df_clean)} training-ready rows")
    return df_clean
