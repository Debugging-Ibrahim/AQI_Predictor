# -*- coding: utf-8 -*-
"""
Model Evaluation Metrics Helper.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """
    Computes RMSE, MAE, and R^2 score for predictions.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- Evaluation Report: {model_name} ---")
    print(f"   Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"   Mean Absolute Error (MAE)       : {mae:.4f}")
    print(f"   R2 Score                       : {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}
