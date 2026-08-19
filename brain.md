# 🧠 AQI Predictor — Project Brain

> **Last Updated:** 2026-08-18
> **City:** Faisalabad, Pakistan (31.4187°N, 73.0791°E)
> **Goal:** Predict Air Quality Index (US AQI) for the next 3 days (72 hours)

---

## 📌 Project Overview

An end-to-end ML system that fetches real-time weather and air quality data, engineers features, trains forecasting models, and serves 3-day AQI predictions through an interactive dashboard. Uses Hopsworks as the feature store and model registry.

---

## ✅ What's Been Done

### 1. Data Fetching (`fetchdata.py`)
- Fetches hourly weather data from Open-Meteo Forecast API
- Fetches hourly air quality data from Open-Meteo Air Quality API
- Merges both datasets on timestamp, tagged with city name
- **Weather fields (11):** temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature, precipitation, rain, surface_pressure, cloud_cover, wind_speed_10m, wind_direction_10m, wind_gusts_10m
- **AQ fields (9):** pm2_5, pm10, ozone, nitrogen_dioxide, sulphur_dioxide, carbon_monoxide, dust, aerosol_optical_depth, us_aqi

### 2. Feature Engineering (`fetchdata.py` → `engineer_features()`)
Features are organized into a tiered system based on atmospheric physics:

#### Cyclic Time Features (6 features)
- hour_sin/cos, day_sin/cos, month_sin/cos

#### Must-Have: AQI & Particulates
| Feature | Lags | Rolling Stats |
|---------|------|---------------|
| us_aqi | 1h, 6h, 24h, 48h, 72h | 24h mean, 72h mean, 1h change rate |
| pm2_5 | 1h, 24h, 48h, 72h | 24h mean |
| pm10 | 1h, 24h | — |

#### Should-Have: Weather Context
| Feature | Lags | Rolling Stats |
|---------|------|---------------|
| temperature_2m | 1h, 24h | — |
| wind_speed_10m | 1h, 6h | — |
| relative_humidity_2m | 1h | — |
| surface_pressure | 1h | — |
| precipitation | — | 6h & 24h cumulative sum |
| rain | — | 6h & 24h cumulative sum |

**Total engineered features: ~45 columns** (20 raw + 6 cyclic + 12 lags + 3 rolling means + 1 change rate + 4 rolling sums)


## 🔲 What's Left (Pipeline Stages)

### Stage 2: Historical Data Backfill `[ ]`
- Use Open-Meteo Archive API to pull 1-2 years of hourly historical data
- Run `engineer_features()` on the full historical dataset
- Generate the training dataset (target: us_aqi at time T+1h through T+72h)
- **Key:** Need to define the target variable clearly — multi-step forecasting (predict next 72 hours)

### Stage 3: Feature Store (Hopsworks) `[ ]`
- Create feature groups in Hopsworks for weather + AQ + engineered features
- Push historical backfill data as the initial feature group
- Set up the hourly feature pipeline to append new rows

### Stage 4: Training Pipeline `[ ]`
- Fetch training data from Hopsworks feature store
- Experiment with models:
  - Random Forest (baseline)
  - XGBoost / LightGBM (primary)
  - Ridge Regression
  - TensorFlow / PyTorch (deep learning — LSTM, Transformer)
- Evaluate with RMSE, MAE, R² metrics
- **Run SHAP analysis after first model trains** → use results to prune/add features
- Store best model in Hopsworks Model Registry

### Stage 5: CI/CD Automation `[ ]`
- Feature pipeline runs **every hour** (fetch + engineer + push to Hopsworks)
- Training pipeline runs **daily** (retrain on latest data)
- Tools: Apache Airflow, GitHub Actions, or similar

### Stage 6: Web Dashboard `[ ]`
- Load model + latest features from Hopsworks
- Compute real-time predictions for next 3 days (72 hours)
- Interactive dashboard with Streamlit/Gradio + Flask/FastAPI backend
- Display AQI forecast, trends, and confidence intervals

### Stage 7: Advanced Analytics `[ ]`
- EDA to identify seasonal trends, pollution hotspots
- SHAP / LIME for feature importance explanations (visual)
- Alerts for hazardous AQI levels (>150)
- Support multiple forecasting model comparisons

---


