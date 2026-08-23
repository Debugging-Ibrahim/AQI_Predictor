# -*- coding: utf-8 -*-
"""
Supabase Database Access Module with 3-Year Historical Fetching and Deduplication Guard.
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from config.settings import SUPABASE_TABLE_NAME

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables / .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_all_features_from_supabase():
    """
    Fetches all historical feature rows from Supabase 'aqi_features' table
    and returns them as a clean Pandas DataFrame for ML training.
    """
    print(f"Querying dataset from Supabase table '{SUPABASE_TABLE_NAME}'...")
    
    all_data = []
    step = 1000
    start = 0

    while True:
        res = supabase.table(SUPABASE_TABLE_NAME)\
                      .select("*")\
                      .order("timestamp", desc=False)\
                      .range(start, start + step - 1)\
                      .execute()
        
        batch = res.data
        if not batch:
            break
        
        all_data.extend(batch)
        if len(batch) < step:
            break
        start += step

    df = pd.DataFrame(all_data)
    if df.empty:
        raise RuntimeError(f"Table '{SUPABASE_TABLE_NAME}' is empty in Supabase!")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"Fetched {len(df)} total rows from Supabase.")
    return df


def fetch_existing_timestamps_from_supabase():
    """
    Fetches set of existing ISO timestamps from Supabase 'aqi_features' table
    and normalizes them as UTC pd.Timestamp objects.
    """
    print(f"Checking existing timestamps in Supabase '{SUPABASE_TABLE_NAME}'...")
    existing = set()
    step = 1000
    start = 0

    while True:
        res = supabase.table(SUPABASE_TABLE_NAME)\
                      .select("timestamp")\
                      .range(start, start + step - 1)\
                      .execute()
        batch = res.data
        if not batch:
            break
        for row in batch:
            if row.get("timestamp"):
                existing.add(pd.to_datetime(row["timestamp"], utc=True))
        if len(batch) < step:
            break
        start += step

    print(f" Found {len(existing)} existing timestamps in Supabase.")
    return existing


def upload_dataframe_to_supabase(df, batch_size=1000, deduplicate=True):
    """
    Formats timestamps and upserts a Pandas DataFrame into Supabase in batches.
    If deduplicate=True, filters out timestamps that already exist in Supabase.
    """
    if df.empty:
        print("Empty dataframe provided for Supabase upload.")
        return

    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)

    if deduplicate:
        existing_timestamps = fetch_existing_timestamps_from_supabase()
        before_count = len(df)
        df = df[~df["timestamp_dt"].isin(existing_timestamps)].reset_index(drop=True)
        after_count = len(df)
        print(f" Deduplication Filter: {before_count} fetched rows -> {after_count} NEW unique rows to upload.")
        
        if df.empty:
            print(" SUCCESS! No new timestamps to upload. Supabase is already up to date.")
            return

    # Filter columns to only upload columns present in Supabase table schema
    known_db_cols = [
        'timestamp', 'city', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'precipitation', 'rain', 'surface_pressure', 'cloud_cover',
        'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m', 'pm2_5', 'pm10', 'ozone',
        'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide', 'dust', 'aerosol_optical_depth',
        'us_aqi', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'us_aqi_lag_1h', 'us_aqi_lag_6h', 'us_aqi_lag_24h', 'us_aqi_lag_48h', 'us_aqi_lag_72h',
        'us_aqi_rolling_mean_24h', 'us_aqi_rolling_mean_72h', 'aqi_change_rate_1h',
        'pm2_5_lag_1h', 'pm2_5_lag_24h', 'pm2_5_lag_48h', 'pm2_5_lag_72h', 'pm2_5_rolling_mean_24h',
        'pm10_lag_1h', 'pm10_lag_24h', 'temperature_2m_lag_1h', 'temperature_2m_lag_24h',
        'wind_speed_10m_lag_1h', 'wind_speed_10m_lag_6h', 'relative_humidity_2m_lag_1h',
        'surface_pressure_lag_1h', 'precipitation_rolling_sum_6h', 'precipitation_rolling_sum_24h',
        'rain_rolling_sum_6h', 'rain_rolling_sum_24h', 'aqi_day1', 'aqi_day2', 'aqi_day3'
    ]
    
    df_to_upload = df[[c for c in df.columns if c in known_db_cols]].copy()
    df_to_upload["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    records = df_to_upload.replace({np.nan: None}).to_dict(orient="records")
    total_records = len(records)

    print(f"Uploading {total_records} new records to Supabase in batches of {batch_size}...")
    for i in range(0, total_records, batch_size):
        batch = records[i:i + batch_size]
        supabase.table(SUPABASE_TABLE_NAME).upsert(batch).execute()
        print(f"   Batch {i // batch_size + 1}/{(total_records + batch_size - 1) // batch_size} inserted ({len(batch)} rows)")

    print(f" SUCCESS! {total_records} new records uploaded to '{SUPABASE_TABLE_NAME}'.")


def run_historical_backfill_upload(start_date=None):
    """
    Fetches historical backfill features for 3 years from fetcher module
    and uploads NEW records into Supabase with automatic deduplication.
    """
    from src.data.fetcher import run_backfill
    if start_date is None:
        start_date = (pd.Timestamp.utcnow() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")

    print(f"Starting 3-Year Historical Backfill Upload ({start_date} to today)...")
    df = run_backfill(start_date=start_date)
    upload_dataframe_to_supabase(df, batch_size=1000, deduplicate=True)


if __name__ == "__main__":
    run_historical_backfill_upload(start_date=None)
