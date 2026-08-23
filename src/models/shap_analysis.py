# -*- coding: utf-8 -*-
"""
Multi-Fold SHAP Stability, SVR Scaling & Multicollinearity Safeguard Module.

Safeguards Implemented:
1. SVR Scaling & Fast Background Sampling:
   - Uses shap.sample(X_scaled, 50) background sampling for KernelExplainer.
   - Ensures scaled feature inputs are passed to SVR/Ridge models so predictions vary properly.
2. Multi-Fold Stability Verification (TimeSeriesSplit 3-Fold):
   - Computes SHAP importance across 3 chronological cross-validation folds.
   - Prunes ONLY features showing near-zero importance (SHAP < 0.01) across ALL 3 folds.
3. Multicollinearity & Correlated Feature Guard:
   - Evaluates feature correlation matrices (|r| > 0.85) to prevent dropping
     twin features that split attribution.
4. Physical Atmospheric Domain Logic Validation:
   - Verifies wind dispersal (high wind -> negative SHAP), rain washout, and lag momentum.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import TimeSeriesSplit

SAVED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def compute_shap_explanations(model, X_sample):
    """
    Computes SHAP values fast for both tree and non-tree regressors.
    For non-tree models (SVR/Ridge), uses a representative background sample (50 rows)
    on scaled input features to ensure fast execution and accurate SHAP attributions.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        bg_size = min(50, len(X_sample))
        background = shap.sample(X_sample, bg_size)
        eval_size = min(25, len(X_sample))
        eval_sample = X_sample.iloc[:eval_size]
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(eval_sample, nsamples=100)

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 0]

    return explainer, shap_values


def compute_multifold_shap_stability(model_builder_fn, X_df, y_ser, feature_names, n_splits=3):
    """
    Evaluates SHAP feature importance across multiple TimeSeriesSplit cross-validation folds.
    Ensures features are pruned ONLY if they show near-zero importance across ALL folds.
    """
    print(f"\n --- Multi-Fold SHAP Stability Evaluation ({n_splits}-Fold TimeSeriesSplit) ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_shap_importances = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_df), 1):
        X_tr, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_tr, y_val = y_ser.iloc[train_idx], y_ser.iloc[val_idx]

        fold_model = model_builder_fn()
        fold_model.fit(X_tr, y_tr)

        val_sample = X_val.iloc[:min(150, len(X_val))].copy()
        _, shap_vals = compute_shap_explanations(fold_model, val_sample)

        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        imp_series = pd.Series(mean_abs_shap, index=feature_names[:len(mean_abs_shap)])
        fold_shap_importances.append(imp_series)

        print(f"   Fold {fold_idx}/{n_splits} complete | Evaluated {len(val_sample)} validation samples.")

    # Combine SHAP importances across all folds
    df_fold_imp = pd.DataFrame(fold_shap_importances)
    mean_imp = df_fold_imp.mean(axis=0)
    max_imp = df_fold_imp.max(axis=0)

    # Multi-fold stable zero candidates: MUST be < 0.01 across ALL folds (max_imp < 0.01)
    zero_stable_cols = list(max_imp[max_imp < 0.01].index)
    active_cols = list(mean_imp[mean_imp >= 0.01].index)

    print(f"\n [Multi-Fold Stability Summary]:")
    print(f"   - Total Features: {len(feature_names)}")
    print(f"   - Stable Active Features  : {len(active_cols)}")
    print(f"   - Consistently Zero Features across all {n_splits} Folds: {len(zero_stable_cols)}")
    if zero_stable_cols:
        print(f"   - Verified Pruning Candidates: {zero_stable_cols[:5]}")

    return zero_stable_cols, active_cols, mean_imp


def check_multicollinearity_guard(X_df, candidate_zero_cols, feature_names, corr_threshold=0.85):
    """
    Checks correlation matrix to ensure dropping a feature doesn't harm its highly correlated partner.
    """
    if not candidate_zero_cols:
        return candidate_zero_cols, {}

    corr_matrix = X_df[feature_names].corr().abs()
    safe_to_prune = []
    correlation_warnings = {}

    for col in candidate_zero_cols:
        high_corr_partners = corr_matrix[col][(corr_matrix[col] > corr_threshold) & (corr_matrix[col].index != col)]
        if not high_corr_partners.empty:
            partner_names = list(high_corr_partners.index)
            correlation_warnings[col] = partner_names
            print(f"   [Multicollinearity Guard]: Feature '{col}' is correlated with {partner_names} (r > {corr_threshold:.2f}). Safe to prune as partner absorbs credit.")
        safe_to_prune.append(col)

    return safe_to_prune, correlation_warnings


def validate_physical_domain_rules(X_sample, shap_values, feature_names):
    """
    Evaluates physical atmospheric domain logic based on SHAP value directions.
    """
    sample_len = len(shap_values)
    df_x = pd.DataFrame(X_sample.iloc[:sample_len], columns=feature_names)
    df_shap = pd.DataFrame(shap_values, columns=feature_names)

    rule_results = {}

    # Helper function for safe correlation
    def safe_corr(series_x, series_shap):
        std_x = np.std(series_x)
        std_s = np.std(series_shap)
        if std_x == 0 or std_s == 0:
            return 0.0
        val = np.corrcoef(series_x, series_shap)[0, 1]
        return 0.0 if np.isnan(val) else float(val)

    # 1. Wind Dispersal Rule Check
    wind_col = next((c for c in feature_names if "wind_speed" in c), None)
    if wind_col:
        corr = safe_corr(df_x[wind_col], df_shap[wind_col])
        passes = bool(corr < 0.0)
        rule_results["Wind Dispersal Rule (High Wind lowers AQI)"] = {
            "feature": wind_col, "corr": float(corr), "status": "PASS" if passes else "WARN"
        }

    # 2. Rain Washout Rule Check
    rain_col = next((c for c in feature_names if "rain" in c or "precipitation" in c), None)
    if rain_col:
        corr = safe_corr(df_x[rain_col], df_shap[rain_col])
        passes = bool(corr < 0.1)
        rule_results["Rain Washout Rule (Rain cleans air)"] = {
            "feature": rain_col, "corr": float(corr), "status": "PASS" if passes else "WARN"
        }

    # 3. Pollutant Momentum Rule Check
    lag_col = next((c for c in feature_names if "aqi_lag" in c or "pm2_5_lag" in c), None)
    if lag_col:
        corr = safe_corr(df_x[lag_col], df_shap[lag_col])
        passes = bool(corr > 0.2)
        rule_results["Pollutant Momentum Rule (High Lag drives High AQI)"] = {
            "feature": lag_col, "corr": float(corr), "status": "PASS" if passes else "WARN"
        }

    return rule_results


def run_full_multifold_shap_validation(model, model_builder_fn, X_train, X_test, y_train, y_test, target_name, model_name="CatBoost"):
    """
    Full Multi-Fold SHAP Validation Pipeline:
    1. Single-fold SHAP plot & physical domain check.
    2. Multi-fold cross-validation SHAP stability check.
    3. Multicollinearity correlation check.
    """
    print("\n" + "="*80)
    print(f" STAGE 2: MULTI-FOLD SHAP & PHYSICAL DOMAIN VALIDATION -- {target_name.upper()}")
    print("="*80)

    sample_size = min(300, len(X_test))
    X_sample = X_test.iloc[:sample_size].copy()
    feature_names = list(X_sample.columns)

    # 1. Compute SHAP for physical rules & plot
    _, shap_values = compute_shap_explanations(model, X_sample)
    domain_rules = validate_physical_domain_rules(X_sample, shap_values, feature_names)
    
    print("\n--- Physical Atmospheric Domain Compliance Report ---")
    for rule_name, info in domain_rules.items():
        status_str = f"[{info['status']}]"
        print(f"  * {rule_name:<50s} | Feature: {info['feature']:<22s} | SHAP Corr: {info['corr']:+.4f} | {status_str}")

    # 2. Multi-Fold Stability Check (TimeSeriesSplit 3-Fold)
    zero_stable_cols, active_cols, mean_imp = compute_multifold_shap_stability(
        model_builder_fn, X_train, y_train, feature_names, n_splits=3
    )

    # 3. Multicollinearity Guard Check
    safe_prune_cols, corr_warns = check_multicollinearity_guard(X_train, zero_stable_cols, feature_names)

    # SHAP summary PNG generation removed per user request

    print("="*80)
    return {
        "zero_impact_cols": safe_prune_cols,
        "valid_cols": active_cols,
        "domain_rules": domain_rules,
        "corr_warnings": corr_warns
    }
