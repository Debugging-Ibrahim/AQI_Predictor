# -*- coding: utf-8 -*-
"""
Daily Multi-Model Retraining Entrypoint.

Executes standardized 4-model retraining pipeline:
- Day 1 SVR (StandardScaler + Exponential Recency Weighting)
- Day 2 SVR (RobustScaler + 2.5x Smog Penalty Weighting)
- Day 3 CatBoost (Standard Fit, Tree-Based, No Scaler)
- Pattern Recognition Engine (57-Feature Vector Re-Indexing)
- Saves model binaries and scalers to saved_models/
- Updates report.md
"""

import sys
import os

from src.models.standardized_pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
