# -*- coding: utf-8 -*-
"""
Supabase Database Access Module.
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
    
    # Supabase PostgREST default limit is 1000, so we paginate to retrieve all rows
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


def upload_dataframe_to_supabase(df, batch_size=1000):
    """
    Formats timestamps and upserts a Pandas DataFrame into Supabase in batches.
    """
    if df.empty:
        print("Empty dataframe provided for Supabase upload.")
        return

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    records = df.replace({np.nan: None}).to_dict(orient="records")
    total_records = len(records)

    print(f"Uploading {total_records} records to Supabase in batches of {batch_size}...")
    for i in range(0, total_records, batch_size):
        batch = records[i:i + batch_size]
        supabase.table(SUPABASE_TABLE_NAME).upsert(batch).execute()
        print(f"   Batch {i // batch_size + 1}/{(total_records + batch_size - 1) // batch_size} inserted ({len(batch)} rows)")

    print(f"SUCCESS! {total_records} records uploaded to '{SUPABASE_TABLE_NAME}'.")


def run_historical_backfill_upload(start_date="2025-01-01"):
    """
    Fetches historical backfill features from fetcher module
    and uploads them into Supabase.
    """
    from src.data.fetcher import run_backfill
    print(f"Starting Historical Backfill Upload ({start_date} to today)...")
    df = run_backfill(start_date=start_date)
    upload_dataframe_to_supabase(df, batch_size=1000)


if __name__ == "__main__":
    run_historical_backfill_upload(start_date="2025-01-01")
