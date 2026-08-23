# -*- coding: utf-8 -*-
"""
Production AQI Forecasting Inference Pipeline (Flask / FastAPI Backend Ready).

Inference Safety Protocol:
1. Loads fitted production scalers ('saved_models/scaler_day1.joblib', 'saved_models/scaler_day2.joblib', etc.).
2. Loads horizon-specific trained winning models ('saved_models/best_aqi_day1.joblib', etc.).
3. Applies 'scaler.transform()' on incoming live feature vectors (NEVER fit_transform).
4. Supports single-row and batch inference.
"""

import os
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

_LOADED_SCALERS = {}
_LOADED_MODELS = {}


def load_production_artifacts():
    """
    Loads saved scalers and model artifacts into memory once for high-throughput inference.
    """
    global _LOADED_SCALERS, _LOADED_MODELS

    for horizon in ["aqi_day1", "aqi_day2", "aqi_day3"]:
        # Load Scaler
        h_short = horizon.replace("aqi_", "")
        scaler_path = os.path.join(SAVED_MODELS_DIR, f"scaler_{h_short}.joblib")
        fallback_scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.joblib")
        
        if os.path.exists(scaler_path):
            _LOADED_SCALERS[horizon] = joblib.load(scaler_path)
        elif os.path.exists(fallback_scaler_path):
            _LOADED_SCALERS[horizon] = joblib.load(fallback_scaler_path)

        # Load Model
        model_path = os.path.join(SAVED_MODELS_DIR, f"best_{horizon}.joblib")
        if os.path.exists(model_path):
            _LOADED_MODELS[horizon] = joblib.load(model_path)
        else:
            print(f" Warning: Model artifact '{model_path}' not found.")

    return _LOADED_SCALERS, _LOADED_MODELS


def predict_aqi_horizon(raw_feature_df, horizon="aqi_day1"):
    """
    Performs production inference for a specified forecast horizon.
    
    Parameters:
        raw_feature_df (pd.DataFrame): Incoming unscaled feature DataFrame.
        horizon (str): 'aqi_day1', 'aqi_day2', or 'aqi_day3'.
        
    Returns:
        np.ndarray: Predicted AQI values.
    """
    global _LOADED_SCALERS, _LOADED_MODELS

    if horizon not in _LOADED_MODELS or horizon not in _LOADED_SCALERS:
        load_production_artifacts()

    model = _LOADED_MODELS[horizon]
    model_type = type(model).__name__

    # SVR and Ridge models require scaled features
    if "SVR" in model_type or "Ridge" in model_type:
        scaler = _LOADED_SCALERS.get(horizon, None)
        if scaler is None:
            raise FileNotFoundError(f"No fitted scaler found for horizon '{horizon}'.")
        scaled_features = scaler.transform(raw_feature_df.fillna(0))
        predictions = model.predict(scaled_features)
    else:
        # Tree-based models (CatBoost, LightGBM, XGBoost, Random Forest) predict directly on feature DF
        predictions = model.predict(raw_feature_df)

    return predictions


if __name__ == "__main__":
    print("Testing Production Inference Pipeline Safety...")
    scalers, models = load_production_artifacts()
    print(f" Loaded Scalers: {list(scalers.keys())}")
    print(f" Loaded Horizon Models: {list(models.keys())}")
