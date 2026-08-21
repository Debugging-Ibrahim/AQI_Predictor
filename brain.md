# 🧠 AQI Predictor — Project Brain & Master Log

> **Last Updated:** 2026-08-19  
> **Location:** Faisalabad, Pakistan (31.4187°N, 73.0791°E)  
> **Goal:** 3-Day (24h, 48h, 72h) US AQI Multi-Horizon Forecasting  
> **Primary Storage:** Supabase PostgreSQL (`aqi_features` table)  
> **Repository:** `Debugging-Ibrahim/AQI_Predictor`  

---

## 📌 System Architecture Overview

An end-to-end Machine Learning system that fetches hourly weather and air quality data from Open-Meteo, engineers 56 physical and temporal features, stores processed datasets in **Supabase PostgreSQL**, trains multi-horizon **XGBoost Regressors**, and serves real-time forecasts via an interactive **Streamlit Dashboard**.

```
[ Open-Meteo APIs ]
        │
        ▼
[ src.data.fetcher ] ── (56 Feature Engineering Engine)
        │
        ▼
[ Supabase PostgreSQL ] ── (aqi_features Table: 14,160 Training Rows)
        │
        ▼
[ src.models.train ] ── (Chronological Split + XGBoost Regressors)
        │
        ▼
[ saved_models/ ] ── (xgb_aqi_day1.joblib, xgb_aqi_day2.joblib, xgb_aqi_day3.joblib)
        │
        ▼
[ Streamlit Dashboard & GitHub Actions CI/CD ] (Upcoming)
```

---

## ✅ Completed Milestones & Detailed Breakdown

### 1. Professional Modular Project Architecture `[x]`
Restructured the repository into a production-grade, modular Python architecture:

```text
AQI_Predictor/
├── config/
│   └── settings.py             # Lat/Lon, API URLs, feature field lists
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Open-Meteo fetcher & 56-feature engineering engine
│   │   └── supabase_db.py      # Supabase queries & backfill upload pipeline
│   ├── models/
│   │   └── train.py            # XGBoost multi-target training pipeline
│   └── utils/
│       └── metrics.py          # RMSE, MAE, R² evaluation metrics
├── saved_models/               # Exported model artifacts (.joblib)
│   ├── xgb_aqi_day1.joblib     # 24-hour forecast model
│   ├── xgb_aqi_day2.joblib     # 48-hour forecast model
│   └── xgb_aqi_day3.joblib     # 72-hour forecast model
├── create_table.sql            # Supabase DDL SQL script
├── requirements.txt            # Python dependencies (xgboost, supabase, etc.)
├── .env                        # Environment variables (SUPABASE_URL, SUPABASE_KEY)
└── .gitignore                  # Git tracking rules
```

---

### 2. Complete 56-Feature Engineering Breakdown `[x]`
Implemented in `src/data/fetcher.py`. All features are engineered using backward-looking windows to ensure **zero data leakage**:

| Category | Count | Column Names | Description |
|---|---|---|---|
| **Primary Keys** | 2 | `timestamp`, `city` | UTC Timestamps (`TIMESTAMPTZ` primary key) & City tag |
| **Raw Weather** | 11 | `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `apparent_temperature`, `precipitation`, `rain`, `surface_pressure`, `cloud_cover`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m` | Hourly surface atmospheric readings from Open-Meteo Archive API |
| **Raw Pollutants** | 9 | `pm2_5`, `pm10`, `ozone`, `nitrogen_dioxide`, `sulphur_dioxide`, `carbon_monoxide`, `dust`, `aerosol_optical_depth`, `us_aqi` | Hourly pollutant concentrations & calculated US AQI index |
| **Cyclic Time** | 6 | `hour_sin`, `hour_cos`, `day_sin`, `day_cos`, `month_sin`, `month_cos` | Sine/Cosine trigonometric transformations for 24h, 7-day, and 12-month cycles |
| **AQI Lags & Rolling** | 8 | `us_aqi_lag_1h`, `us_aqi_lag_6h`, `us_aqi_lag_24h`, `us_aqi_lag_48h`, `us_aqi_lag_72h`, `us_aqi_rolling_mean_24h`, `us_aqi_rolling_mean_72h`, `aqi_change_rate_1h` | Historical AQI trends, rolling baselines, and 1h change rate |
| **$\text{PM}_{2.5}$ & $\text{PM}_{10}$ Lags** | 7 | `pm2_5_lag_1h`, `pm2_5_lag_24h`, `pm2_5_lag_48h`, `pm2_5_lag_72h`, `pm2_5_rolling_mean_24h`, `pm10_lag_1h`, `pm10_lag_24h` | Particulate matter historical momentum indicators |
| **Weather Lags & Sums** | 10 | `temperature_2m_lag_1h`, `temperature_2m_lag_24h`, `wind_speed_10m_lag_1h`, `wind_speed_10m_lag_6h`, `relative_humidity_2m_lag_1h`, `surface_pressure_lag_1h`, `precipitation_rolling_sum_6h`, `precipitation_rolling_sum_24h`, `rain_rolling_sum_6h`, `rain_rolling_sum_24h` | Temperature, wind, pressure lags & cumulative rainfall sums |
| **Target Variables** | 3 | `aqi_day1`, `aqi_day2`, `aqi_day3` | Supervised labels: Actual recorded AQI at $T+24\text{h}$, $T+48\text{h}$, and $T+72\text{h}$ |

---

### 3. Supabase Storage & Historical Backfill Upload `[x]`
* **Database Schema:** Created PostgreSQL table `aqi_features` via `create_table.sql` with `timestamp` as `PRIMARY KEY`.
* **Backfill Execution:** `python -m src.data.supabase_db` pulled **14,304 raw hourly records** (Jan 1, 2025 to Aug 19, 2026).
* **Data Cleaning:** Cleaned initial 72h lag NaNs and final 72h target NaNs $\rightarrow$ **14,160 training-ready rows**.
* **Batch Upsert:** Uploaded to Supabase in 15 batches of 1,000 records.
* **RLS Configuration:** Disabled Row Level Security on `aqi_features` table to allow API upserts.

---

### 4. Machine Learning Model Training `[x]`
Implemented in `src/models/train.py`:

* **Data Leakage Prevention:** Performed **Chronological Time-Series Train/Test Split**:
  * **Train Set (80%):** 11,328 rows (`2025-01-04` to `2026-04-20`)
  * **Test Set (20%):** 2,832 rows (`2026-04-21` to `2026-08-16`)
* **Models Trained:** 3 XGBoost Regressors (`n_estimators=300, learning_rate=0.03, max_depth=6`).
* **Evaluation Metrics:**

| Model Target | Forecast Horizon | Test RMSE | Test MAE | Test $R^2$ | Saved Artifact |
|---|---|---|---|---|---|
| `aqi_day1` | **Day +1 (24h Out)** | **21.62** | **16.87** | **0.3940** | `saved_models/xgb_aqi_day1.joblib` |
| `aqi_day2` | **Day +2 (48h Out)** | **27.41** | **21.12** | **0.0264** | `saved_models/xgb_aqi_day2.joblib` |
| `aqi_day3` | **Day +3 (72h Out)** | **27.93** | **21.79** | **-0.0139** | `saved_models/xgb_aqi_day3.joblib` |

---

## 🔲 What's Left (Next Pipeline Stages)

### Stage 5: Web Application Dashboard (`dashboard/app.py`) `[ ]`
- Build an interactive **Streamlit** dashboard.
- Display current live AQI, weather stats, and gauges for Faisalabad.
- Plot interactive 3-day (24h, 48h, 72h) AQI prediction curves using saved XGBoost models.
- Implement color-coded hazardous AQI alerts (Good, Moderate, Unhealthy, Hazardous).

### Stage 6: Automated CI/CD Incremental Pipeline (`.github/workflows/feature_pipeline.yml`) `[ ]`
- Setup GitHub Actions scheduled workflow (`cron: '0 */3 * * *'`).
- Script logic:
  1. Query `MAX(timestamp)` from Supabase `aqi_features`.
  2. Fetch buffer window (`latest_ts - 72h` to `NOW`) from Open-Meteo.
  3. Compute 56 features.
  4. Filter rows where `timestamp > latest_ts`.
  5. Upsert newly generated hourly records into Supabase.

### Stage 7: Advanced Analytics & Model Explainability `[ ]`
- Integrate **SHAP (SHapley Additive exPlanations)** charts into Streamlit to show feature importances (e.g. how $\text{PM}_{2.5}$ lag or wind speed impacted today's prediction).
- Experiment with LightGBM and hyperparameter tuning to boost Day 2 and Day 3 $R^2$ scores.

---

## 💡 How to Run the Pipeline

```powershell
# 1. Upload Historical Backfill to Supabase
& D:/Anaconda/envs/ds_env/python.exe -m src.data.supabase_db

# 2. Train Models and Export to saved_models/
& D:/Anaconda/envs/ds_env/python.exe -m src.models.train
```
