# -*- coding: utf-8 -*-
"""
Model Evaluation Metrics & Benchmark Helper.

Includes functions to compute RMSE, MAE, R^2, check performance against
the Naive Persistence Baseline (AQI_tomorrow = AQI_today), and verify
compliance with horizon-specific benchmark criteria.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Benchmark targets defined per evaluation guidelines
BENCHMARK_TARGETS = {
    "aqi_day1": {
        "horizon": "Day 1 (24h Ahead)",
        "max_mae": 15.0,
        "max_rmse": 23.0,
        "min_r2": 0.75,
        "target_r2_range": "0.75 - 0.85"
    },
    "aqi_day2": {
        "horizon": "Day 2 (48h Ahead)",
        "max_mae": 22.0,
        "max_rmse": 30.0,
        "min_r2": 0.65,
        "target_r2_range": "0.65 - 0.75"
    },
    "aqi_day3": {
        "horizon": "Day 3 (72h Ahead)",
        "max_mae": 28.0,
        "max_rmse": 38.0,
        "min_r2": 0.50,
        "target_r2_range": "0.50 - 0.65"
    }
}


def compute_metrics(y_true, y_pred):
    """
    Computes RMSE, MAE, and R^2 score for target vs predictions.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """
    Computes and prints basic evaluation metrics.
    """
    m = compute_metrics(y_true, y_pred)
    print(f"\n--- Evaluation Report: {model_name} ---")
    print(f"   Root Mean Squared Error (RMSE) : {m['rmse']:.4f}")
    print(f"   Mean Absolute Error (MAE)       : {m['mae']:.4f}")
    print(f"   R2 Score                       : {m['r2']:.4f}")
    return m


def evaluate_against_benchmarks(target, model_metrics, persistence_metrics):
    """
    Evaluates model metrics against Persistence Baseline and Horizon Benchmarks.
    """
    target_info = BENCHMARK_TARGETS.get(target, {})
    
    # 1. Beating Persistence Baseline (Primary Criterion)
    p_rmse = persistence_metrics["rmse"]
    p_mae = persistence_metrics["mae"]
    
    beats_p_rmse = model_metrics["rmse"] < p_rmse
    beats_p_mae = model_metrics["mae"] < p_mae
    beats_persistence = beats_p_rmse and beats_p_mae
    
    rmse_impr_pct = ((p_rmse - model_metrics["rmse"]) / p_rmse) * 100 if p_rmse > 0 else 0.0
    mae_impr_pct = ((p_mae - model_metrics["mae"]) / p_mae) * 100 if p_mae > 0 else 0.0

    # 2. Horizon Benchmark Compliance
    max_mae = target_info.get("max_mae", 999)
    max_rmse = target_info.get("max_rmse", 999)
    min_r2 = target_info.get("min_r2", -999)
    
    meets_mae = model_metrics["mae"] <= max_mae
    meets_rmse = model_metrics["rmse"] <= max_rmse
    meets_r2 = model_metrics["r2"] >= min_r2
    
    meets_all_benchmarks = beats_persistence and meets_mae and meets_rmse and meets_r2
    
    return {
        "beats_persistence": beats_persistence,
        "rmse_impr_pct": rmse_impr_pct,
        "mae_impr_pct": mae_impr_pct,
        "meets_all_benchmarks": meets_all_benchmarks,
        "meets_mae": meets_mae,
        "meets_rmse": meets_rmse,
        "meets_r2": meets_r2
    }


def print_comparison_table(target, results_dict, persistence_metrics):
    """
    Prints a formatted terminal table comparing all models for a specific horizon.
    """
    target_info = BENCHMARK_TARGETS.get(target, {"horizon": target})
    
    print("\n" + "=" * 90)
    print(f" MODEL EVALUATION & BENCHMARK REPORT -- {target_info.get('horizon', target).upper()}")
    print("=" * 90)
    print(f" Target Targets: MAE < {target_info.get('max_mae')} | RMSE < {target_info.get('max_rmse')} | R2 Range: {target_info.get('target_r2_range')}")
    print("-" * 90)
    print(f"{'Model Algorithm':<22} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8} | {'vs Persist. RMSE':<18} | {'Persistence':<12} | {'Status':<12}")
    print("-" * 90)
    
    # Print Persistence Baseline Row
    print(f"{'Persistence Baseline':<22} | {persistence_metrics['rmse']:<8.2f} | {persistence_metrics['mae']:<8.2f} | {persistence_metrics['r2']:<8.4f} | {'0.00% (Baseline)':<18} | {'BASELINE':<12} | {'N/A':<12}")
    print("-" * 90)

    for model_name, metrics in results_dict.items():
        if model_name == "Persistence":
            continue
            
        bench = evaluate_against_benchmarks(target, metrics, persistence_metrics)
        impr_str = f"-{bench['rmse_impr_pct']:.2f}%" if bench['rmse_impr_pct'] >= 0 else f"+{abs(bench['rmse_impr_pct']):.2f}%"
        
        persist_status = "PASS" if bench['beats_persistence'] else "FAIL"
        bench_status = "MET" if bench['meets_all_benchmarks'] else ("BEATS BASELINE" if bench['beats_persistence'] else "REJECTED")
        
        print(f"{model_name:<22} | {metrics['rmse']:<8.2f} | {metrics['mae']:<8.2f} | {metrics['r2']:<8.4f} | {impr_str:<18} | {persist_status:<12} | {bench_status:<12}")

    print("=" * 90)
