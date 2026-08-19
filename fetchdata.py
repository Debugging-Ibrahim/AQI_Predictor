import requests
import pandas as pd
import numpy as np

LAT, LON = 31.4187, 73.0791
CITY_NAME = "Faisalabad"

# Cleaned, curated variable lists
WEATHER_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"
]

AIR_QUALITY_FIELDS = [
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "sulphur_dioxide", "carbon_monoxide", "dust", "aerosol_optical_depth", "us_aqi"
]

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_hourly(base_url, fields, past_days=0, start_date=None, end_date=None):
    """Generic fetch for Open-Meteo endpoints."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(fields),
        "timezone": "auto",
    }
    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_days"] = past_days
        params["forecast_days"] = 1  # Only fetch up to present time

    r = requests.get(base_url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["hourly"]


def hourly_json_to_rows(hourly_json):
    """Generic JSON to rows parser."""
    field_names = [k for k in hourly_json.keys() if k != "time"]
    rows = []
    for i, ts in enumerate(hourly_json["time"]):
        row = {"timestamp": ts}
        for field in field_names:
            row[field] = hourly_json[field][i]
        rows.append(row)
    return rows


def merge_on_timestamp(*row_lists):
    """Merges weather and air quality lists on timestamp."""
    merged = {}
    for rows in row_lists:
        for row in rows:
            ts = row["timestamp"]
            merged.setdefault(ts, {"timestamp": ts, "city": CITY_NAME})
            merged[ts].update({k: v for k, v in row.items() if k != "timestamp"})
    return sorted(merged.values(), key=lambda r: r["timestamp"])


def engineer_features(df):
    """
    Adds Cyclic Time Encodings, Lag Features, Rolling Stats, and Derivatives.

    Feature tiers (lean set — expand later based on SHAP importance):
      Must-Have:   us_aqi, pm2_5, pm10  (lags + rolling stats)
      Should-Have: temperature, wind, humidity, pressure, precipitation/rain
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── 1. Cyclic Time Features ──────────────────────────────────────
    hour = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── 2. MUST-HAVE: us_aqi — Lags + Rolling Stats ─────────────────
    df["us_aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["us_aqi_lag_6h"] = df["us_aqi"].shift(6)
    df["us_aqi_lag_24h"] = df["us_aqi"].shift(24)
    df["us_aqi_lag_48h"] = df["us_aqi"].shift(48)
    df["us_aqi_lag_72h"] = df["us_aqi"].shift(72)
    df["us_aqi_rolling_mean_24h"] = df["us_aqi"].rolling(window=24).mean()
    df["us_aqi_rolling_mean_72h"] = df["us_aqi"].rolling(window=72).mean()
    df["aqi_change_rate_1h"] = df["us_aqi"] - df["us_aqi_lag_1h"]

    # ── 3. MUST-HAVE: pm2_5 — Lags + Rolling Stats ──────────────────
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["pm2_5_lag_48h"] = df["pm2_5"].shift(48)
    df["pm2_5_lag_72h"] = df["pm2_5"].shift(72)
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].rolling(window=24).mean()

    # ── 4. MUST-HAVE: pm10 — Lags ───────────────────────────────────
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_24h"] = df["pm10"].shift(24)

    # ── 5. SHOULD-HAVE: Weather Contextual Lags ─────────────────────
    df["temperature_2m_lag_1h"] = df["temperature_2m"].shift(1)
    df["temperature_2m_lag_24h"] = df["temperature_2m"].shift(24)

    df["wind_speed_10m_lag_1h"] = df["wind_speed_10m"].shift(1)
    df["wind_speed_10m_lag_6h"] = df["wind_speed_10m"].shift(6)

    df["relative_humidity_2m_lag_1h"] = df["relative_humidity_2m"].shift(1)

    df["surface_pressure_lag_1h"] = df["surface_pressure"].shift(1)

    # ── 6. SHOULD-HAVE: Precipitation / Rain — Cumulative Sums ──────
    df["precipitation_rolling_sum_6h"] = df["precipitation"].rolling(window=6).sum()
    df["precipitation_rolling_sum_24h"] = df["precipitation"].rolling(window=24).sum()
    df["rain_rolling_sum_6h"] = df["rain"].rolling(window=6).sum()
    df["rain_rolling_sum_24h"] = df["rain"].rolling(window=24).sum()

    return df


if __name__ == "__main__":
    # Fetch past 7 days to give enough history for 72h lags
    weather_json = fetch_hourly(WEATHER_URL, WEATHER_FIELDS, past_days=7)
    aq_json = fetch_hourly(AQ_URL, AIR_QUALITY_FIELDS, past_days=7)

    raw_rows = merge_on_timestamp(
        hourly_json_to_rows(weather_json),
        hourly_json_to_rows(aq_json)
    )

    # Convert to DataFrame and compute ML features
    df_raw = pd.DataFrame(raw_rows)
    df_features = engineer_features(df_raw)

    # Drop early rows containing NaNs created by 72h lag shifts
    df_clean = df_features.dropna().reset_index(drop=True)

    print(f"Extracted {len(df_raw)} raw hours.")
    print(f"Engineered {len(df_clean.columns)} features ready for Hopsworks.")
    print("\nSample processed row:")
    print(df_clean.iloc[-1].to_dict())