# -*- coding: utf-8 -*-
"""
Model Training Pipeline.

1. Fetches historical engineered dataset from Supabase.
2. Performs chronological time-series train/test split.
3. Trains XGBoost Regressors for Day 1, Day 2, and Day 3 AQI predictions.
4. Evaluates performance (RMSE, MAE, R^2).
5. Exports trained model artifacts to 'saved_models/' directory.
"""

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor

from src.data.supabase_db import fetch_all_features_from_supabase
from src.utils.metrics import evaluate_predictions

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

TARGET_COLS = ["aqi_day1", "aqi_day2", "aqi_day3"]
EXCLUDE_COLS = ["timestamp", "city"] + TARGET_COLS


def prepare_datasets(df, test_ratio=0.2):
    """
    Performs chronological time-series split to prevent data leakage.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    print(f" Chronological Split:")
    print(f"   Train set: {len(X_train)} rows ({train_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {train_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"   Test set : {len(X_test)} rows ({test_df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {test_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')})")
    print(f"   Number of Features: {len(feature_cols)}")

    return X_train, X_test, train_df, test_df, feature_cols


def train_and_evaluate_models():
    """
    Main training workflow for 24h, 48h, and 72h AQI forecasting.
    """
    print("\n" + "="*60)
    print("STARTING AQI MODEL TRAINING PIPELINE")
    print("="*60)

    # Step 1: Fetch data from Supabase
    df = fetch_all_features_from_supabase()

    # Step 2: Split data chronologically
    X_train, X_test, train_df, test_df, feature_cols = prepare_datasets(df, test_ratio=0.2)

    trained_models = {}
    metrics_summary = {}

    # Step 3: Train models for each target
    for target in TARGET_COLS:
        print("\n" + "-"*60)
        print(f"Training Model for Target: '{target}'")
        print("-"*60)

        y_train = train_df[target]
        y_test = test_df[target]

        # Initialize XGBoost Regressor
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predict & Evaluate
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, model_name=f"XGBoost_{target}")

        # Save trained model artifact
        model_filename = f"xgb_{target}.joblib"
        model_path = os.path.join(SAVED_MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"Saved model artifact to: '{model_path}'")

        trained_models[target] = model
        metrics_summary[target] = metrics

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nFinal Models Performance Summary:")
    for target, m in metrics_summary.items():
        print(f"  * {target:10s} -> RMSE: {m['rmse']:.2f} | MAE: {m['mae']:.2f} | R2: {m['r2']:.4f}")
    print("="*60)

    return trained_models, feature_cols


if __name__ == "__main__":
    train_and_evaluate_models()
