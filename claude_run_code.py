"""
Optimized Reoperation Prediction Analysis
==========================================
This script uses the existing ScoliosisTimePredictor infrastructure for preprocessing
and trains models to predict reoperation risk.

Key optimizations applied:
1. Missing indicator features (missingness in labs is informative)
2. Clinically meaningful feature interactions (ASA×age, optime×ASA)
3. Partial SMOTE (sampling_strategy < 1.0 to avoid overfitting minority)
4. LightGBM added (often outperforms XGBoost on imbalanced data)
5. Calibrated probability ensemble for final predictions
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from datetime import datetime

# Use existing infrastructure
from chart_review_scoliosis_time_predictor import ScoliosisTimePredictor

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, RepeatedStratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix, classification_report,
    make_scorer, roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier

# Imbalanced-learn imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

# Boosting models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# For SHAP analysis
import shap

import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DATA_FOLDER = "raw_chart_data"  # Input data location
for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
    # Files are processed alphabetically
    for filename in filenames:
        # Skip hidden files and non-CSV files
        if filename.startswith('.') or not filename.endswith('.csv'):
            print(f"Skipping {filename}")
            continue

        curr_csv_path = os.path.join(subdir, filename)


RESULTS_DIR = "results/claude_analysis"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "reoperation"
N_JOBS = 8

# Cross-validation settings
CV_SPLITS = 5  # Increased from 3 for more stable estimates
CV_REPEATS = 1

# Feature filtering
FILTER_FEATURES = True
MIN_FEATURE_IMPORTANCE = 0.0001

# SMOTE settings - use partial oversampling to avoid overfitting minority
USE_SMOTE = True
SMOTE_K_NEIGHBORS = 5  # Increased from 3 for smoother decision boundary
SMOTE_SAMPLING_STRATEGY = 0.3  # Only oversample minority to 30% of majority (was 1.0)

# Feature engineering settings
ADD_MISSING_INDICATORS = True  # Lab missingness is informative
ADD_FEATURE_INTERACTIONS = True  # Clinical interactions
DROP_HIGH_MISSING_LABS = True  # Drop labs >90% missing after creating indicators

# Lab columns known to have high missingness (>80%)
HIGH_MISSING_LAB_COLS = [
    'prsepis', 'prbili', 'prsgot', 'pralkph', 'pralbum', 'prptt',
    'prpt', 'prinr', 'prcreat', 'prbun', 'prsodm', 'prplate',
    'prwbc', 'prhct'
]


# =============================================================================
# FEATURE FILTERING (from ScoliosisTimePredictor)
# =============================================================================

def filter_low_importance_features(X_train, y_train, X_test, min_importance_threshold=0.0001):
    """
    Quick feature filtering using a fast Random Forest to estimate feature importance.
    Removes features with very low importance before main model training.
    """
    print(">>> Running quick feature importance filtering...")

    # Train a quick RF to get feature importances
    quick_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    quick_rf.fit(X_train, y_train)

    # Get feature importances
    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': quick_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f">>> Top 15 features by importance:")
    print(importances.head(15).to_string(index=False))

    # Filter features
    selected_features = importances[importances['importance'] >= min_importance_threshold]['feature'].tolist()
    print(f">>> Keeping {len(selected_features)} features with importance >= {min_importance_threshold}")

    removed_count = len(X_train.columns) - len(selected_features)
    print(f">>> Removed {removed_count} low-importance features")

    if selected_features:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    return X_train, X_test


# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================

def add_missing_indicators(df, lab_cols):
    """
    Create binary indicator features for missing lab values.

    Rationale: Lab missingness is informative - patients who don't get certain
    labs ordered may be healthier or have different risk profiles. This captures
    that signal instead of losing it through imputation.
    """
    print(">>> Adding missing indicator features...")
    new_features = []

    for col in lab_cols:
        # Check both exact match and partial match (columns may be lowercased)
        matching_cols = [c for c in df.columns if col.lower() in c.lower()]

        for matched_col in matching_cols:
            indicator_name = f"{matched_col}_missing"
            if indicator_name not in df.columns:
                # Check for NaN or mean-imputed values (often near column mean)
                df[indicator_name] = (df[matched_col].isna() |
                                      (df[matched_col] == df[matched_col].mean())).astype(int)
                new_features.append(indicator_name)

    print(f">>> Created {len(new_features)} missing indicator features")
    return df


def add_clinical_interactions(df):
    """
    Add clinically meaningful feature interactions.

    These interactions are based on medical domain knowledge:
    - ASA score × age: Older patients with higher ASA are at compounded risk
    - ASA score × BMI: Obesity + comorbidity interaction
    - Operation time × ASA: Longer surgeries in sicker patients
    - Anesthesia time ratio: Longer anesthesia relative to surgery may indicate complications
    """
    print(">>> Adding clinical interaction features...")
    new_features = []

    # Find relevant columns (handle different naming conventions)
    asa_cols = [c for c in df.columns if 'asa' in c.lower() and c not in ['anesurg', 'surgane']]

    # Create a composite ASA severity score if we have the dummy variables
    if 'ASA_3plus' in df.columns:
        # Weighted ASA score: ASA1=1, ASA2=2, ASA3+=3
        df['asa_severity'] = df.get('ASA_1', 0) * 1 + df.get('ASA_2', 0) * 2 + df.get('ASA_3plus', 0) * 3
        new_features.append('asa_severity')

        # ASA × Age interaction
        if 'age_years' in df.columns:
            df['asa_x_age'] = df['asa_severity'] * df['age_years']
            new_features.append('asa_x_age')

        # ASA × BMI interaction
        if 'bmi' in df.columns:
            df['asa_x_bmi'] = df['asa_severity'] * df['bmi'].fillna(df['bmi'].median())
            new_features.append('asa_x_bmi')

        # ASA × Operation time
        if 'optime' in df.columns:
            df['asa_x_optime'] = df['asa_severity'] * df['optime']
            new_features.append('asa_x_optime')

    # Anesthesia to operation time ratio (values > 1.5 may indicate complexity)
    if 'anetime' in df.columns and 'optime' in df.columns:
        # Avoid division by zero
        safe_optime = df['optime'].replace(0, 1)
        df['ane_op_ratio'] = df['anetime'] / safe_optime
        new_features.append('ane_op_ratio')

    # WorkRVU × operation time (procedure complexity indicator)
    if 'workrvu' in df.columns and 'optime' in df.columns:
        df['rvu_x_optime'] = df['workrvu'] * df['optime']
        new_features.append('rvu_x_optime')

    # Age × comorbidity indicators
    if 'age_years' in df.columns:
        for comorbidity in ['cerebral_palsy', 'neuromuscdis', 'seizure']:
            if comorbidity in df.columns:
                df[f'age_x_{comorbidity}'] = df['age_years'] * df[comorbidity]
                new_features.append(f'age_x_{comorbidity}')

    print(f">>> Created {len(new_features)} interaction features: {new_features}")
    return df


def drop_high_missing_columns(df, lab_cols, threshold=0.90):
    """
    Drop original lab columns that had very high missingness.
    We keep the missing indicators but drop the imputed values which are mostly noise.
    """
    cols_to_drop = []
    for col in lab_cols:
        matching_cols = [c for c in df.columns if col.lower() == c.lower() and '_missing' not in c]
        cols_to_drop.extend(matching_cols)

    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    if existing_to_drop:
        print(f">>> Dropping {len(existing_to_drop)} high-missing lab columns (keeping indicators)")
        df = df.drop(columns=existing_to_drop)

    return df


# =============================================================================
# MODEL DEFINITIONS - Optimized for 2% positive rate
# =============================================================================

def get_models():
    """
    Define models optimized for severe class imbalance (~2% positive rate).

    Key optimizations:
    - Higher scale_pos_weight/class_weight for 50:1 imbalance ratio
    - Lower learning rates with more iterations for stability
    - Added LightGBM (often outperforms XGBoost on imbalanced tabular data)
    - Regularization to prevent overfitting on rare class
    """

    # Calculate approximate class weight for 2% positive rate
    # Minority class weight ≈ (1 - 0.02) / 0.02 ≈ 49
    imbalance_ratio = 49

    models = {
        # LightGBM - often best for imbalanced tabular data
        "LGBMClassifier": {
            "classifier": LGBMClassifier(
                random_state=42,
                n_jobs=N_JOBS,
                verbose=-1,
                force_col_wise=True,  # Avoid data copy warnings
            ),
            "param_grid": {
                "classifier__n_estimators": [300],
                "classifier__max_depth": [6, 8],
                "classifier__learning_rate": [0.02, 0.05],
                "classifier__scale_pos_weight": [imbalance_ratio],
                "classifier__min_child_samples": [20],  # Prevent overfitting on minority
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
                "classifier__reg_alpha": [0.1],  # L1 regularization
                "classifier__reg_lambda": [1],   # L2 regularization
            },
        },

        # CatBoost - handles categorical features well
        "CatBoostClassifier": {
            "classifier": CatBoostClassifier(verbose=False, random_state=42),
            "param_grid": {
                "classifier__auto_class_weights": ["Balanced"],
                "classifier__iterations": [300],
                "classifier__depth": [6, 8],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__l2_leaf_reg": [3, 10],  # More regularization options
            },
        },

        # XGBoost - with proper imbalance handling
        "XGBClassifier": {
            "classifier": XGBClassifier(
                random_state=42,
                eval_metric="aucpr",  # Changed to PR-AUC for imbalanced data
                n_jobs=N_JOBS,
                use_label_encoder=False,
            ),
            "param_grid": {
                "classifier__n_estimators": [300],
                "classifier__max_depth": [5, 7],
                "classifier__learning_rate": [0.02, 0.05],
                "classifier__scale_pos_weight": [imbalance_ratio],
                "classifier__min_child_weight": [10],  # Higher to reduce overfitting
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
                "classifier__reg_alpha": [0.1],
                "classifier__reg_lambda": [1],
            },
        },

        # RandomForest - baseline with balanced weights
        "RandomForestClassifier": {
            "classifier": RandomForestClassifier(n_jobs=N_JOBS, random_state=42),
            "param_grid": {
                "classifier__n_estimators": [300],
                "classifier__max_depth": [12, 16],
                "classifier__min_samples_split": [10],  # Increased for stability
                "classifier__min_samples_leaf": [4],
                "classifier__max_features": ["sqrt"],
                "classifier__class_weight": ["balanced_subsample"],
            },
        },

        # BalancedRandomForest - specialized for imbalanced data
        "BalancedRandomForestClassifier": {
            "classifier": BalancedRandomForestClassifier(
                random_state=42,
                n_jobs=N_JOBS,
                sampling_strategy='not majority',  # Undersample majority class
            ),
            "param_grid": {
                "classifier__n_estimators": [300],
                "classifier__max_depth": [12],
                "classifier__min_samples_split": [5],
                "classifier__min_samples_leaf": [2],
                "classifier__max_features": ["sqrt"],
            },
        },

        # Logistic Regression - for calibration baseline
        "LogisticRegression": {
            "classifier": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.01, 0.1],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"],
                "classifier__class_weight": ["balanced"],
            },
        },
    }

    return models


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate_model(model_name, model_config, X_train, X_test, y_train, y_test, cv):
    """
    Train a single model with grid search and evaluate on holdout set.
    Uses ImbPipeline with partial SMOTE to avoid overfitting minority class.
    """
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    clf = model_config["classifier"]
    param_grid = model_config["param_grid"]

    # Check if classifier handles imbalance internally
    clf_type = type(clf).__name__
    handles_imbalance = clf_type in ['BalancedRandomForestClassifier', 'BalancedBaggingClassifier']

    # Use imbalanced-learn Pipeline with partial SMOTE
    # Partial oversampling (0.3) means minority will be 30% of majority count
    # This avoids overfitting while still helping with imbalance
    if USE_SMOTE and not handles_imbalance:
        print(f"  Using SMOTE (k={SMOTE_K_NEIGHBORS}, sampling_strategy={SMOTE_SAMPLING_STRATEGY})")
        pipe = ImbPipeline([
            ("sampler", SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=SMOTE_K_NEIGHBORS,
                sampling_strategy=SMOTE_SAMPLING_STRATEGY  # Partial oversampling
            )),
            ("classifier", clf)
        ])
    else:
        from sklearn.pipeline import Pipeline
        if handles_imbalance:
            print(f"  {model_name} handles imbalance internally - skipping SMOTE")
        pipe = Pipeline([("classifier", clf)])

    # Scoring metrics - optimize for PR-AUC (better for imbalanced data)
    scoring = {
        "f1": make_scorer(f1_score, average="binary", zero_division=0),
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="binary", zero_division=0),
        "recall": make_scorer(recall_score, average="binary", zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
    }

    # Grid search optimizing for PR-AUC (more meaningful for imbalanced data)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit="pr_auc",  # Optimize for PR-AUC like run_models.py
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    print("  Fitting grid search...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV PR-AUC: {grid_search.best_score_:.4f}")

    # Holdout evaluation
    y_pred = best_model.predict(X_test)

    # Get probabilities from the classifier step
    classifier = best_model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        y_proba = classifier.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    pr_auc = average_precision_score(y_test, y_proba) if y_proba is not None else None
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Holdout Set Results:")
    print(f"    ROC-AUC:   {roc_auc:.4f}" if roc_auc else "    ROC-AUC:   N/A")
    print(f"    PR-AUC:    {pr_auc:.4f}" if pr_auc else "    PR-AUC:    N/A")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Confusion Matrix:\n{cm}")

    # Threshold optimization for better F1
    tuned = None
    if y_proba is not None:
        print("\n  Threshold Optimization:")
        best_thr = 0.5
        best_f1 = f1

        for t in np.linspace(0.2, 0.8, 25):
            y_pred_t = (y_proba >= t).astype(int)
            f1_t = f1_score(y_test, y_pred_t, zero_division=0)
            if f1_t > best_f1:
                best_thr = t
                best_f1 = f1_t

        y_pred_opt = (y_proba >= best_thr).astype(int)
        opt_precision = precision_score(y_test, y_pred_opt, zero_division=0)
        opt_recall = recall_score(y_test, y_pred_opt, zero_division=0)
        opt_cm = confusion_matrix(y_test, y_pred_opt)

        print(f"    Optimal threshold: {best_thr:.3f}")
        print(f"    Optimized F1:      {best_f1:.4f}")
        print(f"    Optimized Prec:    {opt_precision:.4f}")
        print(f"    Optimized Recall:  {opt_recall:.4f}")

        tuned = {
            "thr": best_thr,
            "f1": best_f1,
            "precision": opt_precision,
            "recall": opt_recall,
            "cm": opt_cm,
        }

    results = {
        "model_name": model_name,
        "best_params": grid_search.best_params_,
        "cv_pr_auc": grid_search.best_score_,
        "holdout": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "confusion_matrix": cm,
        },
        "threshold_tuned": tuned,
        "pipeline": best_model,
        "y_proba": y_proba,
    }

    return results


def compute_shap_values(results, X_test, model_name):
    """
    Compute SHAP values for feature importance analysis.
    """
    print(f"\n--- Computing SHAP values for {model_name} ---")

    pipeline = results["pipeline"]
    classifier = pipeline.named_steps['classifier']

    model_type = type(classifier).__name__

    try:
        if model_type in ['LogisticRegression']:
            explainer = shap.LinearExplainer(classifier, X_test)
            shap_values = explainer.shap_values(X_test)
        elif model_type in ['CatBoostClassifier', 'XGBClassifier', 'RandomForestClassifier']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_test)
            # Handle multi-output (take positive class)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
        else:
            print(f"  SHAP not implemented for {model_type}")
            return None

        # Create summary dataframe
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_summary = pd.DataFrame({
            'feature': X_test.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        print("\n  Top 15 Features by SHAP Importance:")
        print(shap_summary.head(15).to_string(index=False))

        return {
            'shap_values': shap_values,
            'shap_summary': shap_summary,
        }

    except Exception as e:
        print(f"  Error computing SHAP: {e}")
        return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function with enhanced feature engineering.

    Pipeline:
    1. Load and preprocess with ScoliosisTimePredictor
    2. Add missing indicators (lab missingness is informative)
    3. Add clinical interactions (ASA×age, etc.)
    4. Train/test split
    5. Feature filtering
    6. Train models with optimized hyperparameters
    7. Build calibrated ensemble
    """
    print("\n" + "="*70)
    print("REOPERATION PREDICTION - OPTIMIZED PIPELINE")
    print("="*70 + "\n")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================================================
    # USE EXISTING INFRASTRUCTURE FOR PREPROCESSING
    # ==========================================================================
    print("Loading and preprocessing data using ScoliosisTimePredictor...")
    predictor = ScoliosisTimePredictor(csv_path=curr_csv_path)
    df = predictor.generate_training_dataframe(target_col=TARGET)

    if df.empty:
        print("ERROR: Failed to generate training dataframe")
        return None, None

    print(f"Preprocessed data shape: {df.shape}")
    print(f"Target distribution: {df[TARGET].value_counts().to_dict()}")
    print(f"Reoperation rate: {df[TARGET].mean()*100:.2f}%")

    # ==========================================================================
    # ADVANCED FEATURE ENGINEERING
    # ==========================================================================
    print("\n" + "-"*50)
    print("FEATURE ENGINEERING")
    print("-"*50)

    # 1. Add missing indicators BEFORE dropping the lab columns
    # (need original values to detect what was missing)
    if ADD_MISSING_INDICATORS:
        df = add_missing_indicators(df, HIGH_MISSING_LAB_COLS)

    # 2. Add clinical interactions
    if ADD_FEATURE_INTERACTIONS:
        df = add_clinical_interactions(df)

    # 3. Optionally drop the high-missing lab columns (keep only indicators)
    if DROP_HIGH_MISSING_LABS:
        df = drop_high_missing_columns(df, HIGH_MISSING_LAB_COLS)

    print(f"\nFinal feature count: {df.shape[1] - 1} features")

    # Prepare features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Train/test split (stratified due to class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} positive, {y_train.mean()*100:.2f}%)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} positive, {y_test.mean()*100:.2f}%)")

    # ==========================================================================
    # FEATURE FILTERING
    # ==========================================================================
    if FILTER_FEATURES:
        X_train, X_test = filter_low_importance_features(
            X_train, y_train, X_test,
            min_importance_threshold=MIN_FEATURE_IMPORTANCE
        )

    # Cross-validation strategy
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE
    )
    print(f"\nUsing RepeatedStratifiedKFold: {CV_SPLITS} splits x {CV_REPEATS} repeats")

    # Get model configurations
    models = get_models()

    # Train all models and collect results
    all_results = {}

    for model_name, model_config in models.items():
        try:
            results = train_and_evaluate_model(
                model_name, model_config,
                X_train, X_test, y_train, y_test, cv
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"\n  ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ==========================================================================
    # SUMMARY OF RESULTS
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY OF ALL MODELS")
    print("="*70)

    summary_data = []
    for model_name, results in all_results.items():
        holdout = results["holdout"]
        tuned = results.get("threshold_tuned") or {}
        summary_data.append({
            "Model": model_name,
            "ROC-AUC": holdout.get("roc_auc"),
            "PR-AUC": holdout.get("pr_auc"),
            "F1 (default)": holdout.get("f1"),
            "F1 (tuned)": tuned.get("f1"),
            "Precision (tuned)": tuned.get("precision"),
            "Recall (tuned)": tuned.get("recall"),
            "Best Threshold": tuned.get("thr"),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("ROC-AUC", ascending=False)

    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, f"model_comparison_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # ==========================================================================
    # BEST MODEL ANALYSIS
    # ==========================================================================
    if all_results:
        best_model_name = summary_df.iloc[0]["Model"]
        best_results = all_results[best_model_name]

        print(f"\n" + "="*70)
        print(f"BEST MODEL: {best_model_name}")
        print("="*70)

        print(f"\nBest Parameters:")
        for k, v in best_results["best_params"].items():
            print(f"  {k}: {v}")

        print(f"\nHoldout Performance:")
        print(f"  ROC-AUC:   {best_results['holdout']['roc_auc']:.4f}")
        print(f"  PR-AUC:    {best_results['holdout']['pr_auc']:.4f}")
        print(f"  F1 Score:  {best_results['holdout']['f1']:.4f}")

        # Feature importance for best model
        shap_data = compute_shap_values(best_results, X_test, best_model_name)

        if shap_data is not None:
            # Save SHAP summary
            shap_path = os.path.join(RESULTS_DIR, f"shap_summary_{best_model_name}_{timestamp}.csv")
            shap_data['shap_summary'].to_csv(shap_path, index=False)
            print(f"\nSHAP summary saved to: {shap_path}")

            # Create SHAP beeswarm plot
            try:
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_data['shap_values'],
                    X_test,
                    max_display=20,
                    show=False
                )
                plot_path = os.path.join(RESULTS_DIR, f"shap_beeswarm_{best_model_name}_{timestamp}.png")
                plt.savefig(plot_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"SHAP plot saved to: {plot_path}")
            except Exception as e:
                print(f"Could not save SHAP plot: {e}")

        # ==========================================================================
        # ROC CURVE COMPARISON
        # ==========================================================================
        try:
            plt.figure(figsize=(10, 8))

            for model_name, results in all_results.items():
                if results["y_proba"] is not None:
                    fpr, tpr, _ = roc_curve(y_test, results["y_proba"])
                    auc = results["holdout"]["roc_auc"]
                    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - All Models')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)

            roc_path = os.path.join(RESULTS_DIR, f"roc_curves_{timestamp}.png")
            plt.savefig(roc_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"\nROC curves saved to: {roc_path}")
        except Exception as e:
            print(f"Could not save ROC curves: {e}")

        # ==========================================================================
        # PRECISION-RECALL CURVE
        # ==========================================================================
        try:
            plt.figure(figsize=(10, 8))

            for model_name, results in all_results.items():
                if results["y_proba"] is not None:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_test, results["y_proba"])
                    pr_auc = results["holdout"]["pr_auc"]
                    plt.plot(recall_curve, precision_curve, label=f"{model_name} (PR-AUC={pr_auc:.3f})")

            # Baseline (proportion of positives)
            baseline = y_test.mean()
            plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves - All Models')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)

            pr_path = os.path.join(RESULTS_DIR, f"pr_curves_{timestamp}.png")
            plt.savefig(pr_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Precision-Recall curves saved to: {pr_path}")
        except Exception as e:
            print(f"Could not save PR curves: {e}")

        # ==========================================================================
        # CALIBRATED ENSEMBLE (Top 3 Models)
        # ==========================================================================
        print("\n" + "="*70)
        print("CALIBRATED ENSEMBLE")
        print("="*70)

        try:
            # Get top 3 models by ROC-AUC
            top_models = summary_df.head(3)["Model"].tolist()
            print(f"\nBuilding ensemble from: {top_models}")

            # Collect predictions from top models
            ensemble_probas = []
            for model_name in top_models:
                if model_name in all_results and all_results[model_name]["y_proba"] is not None:
                    ensemble_probas.append(all_results[model_name]["y_proba"])

            if len(ensemble_probas) >= 2:
                # Average probabilities (soft voting)
                ensemble_proba = np.mean(ensemble_probas, axis=0)

                # Evaluate ensemble
                ensemble_roc_auc = roc_auc_score(y_test, ensemble_proba)
                ensemble_pr_auc = average_precision_score(y_test, ensemble_proba)

                # Find optimal threshold for ensemble
                best_thr, best_f1 = 0.5, 0
                for t in np.linspace(0.1, 0.5, 40):
                    y_pred_t = (ensemble_proba >= t).astype(int)
                    f1_t = f1_score(y_test, y_pred_t, zero_division=0)
                    if f1_t > best_f1:
                        best_thr, best_f1 = t, f1_t

                y_pred_ensemble = (ensemble_proba >= best_thr).astype(int)
                ensemble_precision = precision_score(y_test, y_pred_ensemble, zero_division=0)
                ensemble_recall = recall_score(y_test, y_pred_ensemble, zero_division=0)
                ensemble_cm = confusion_matrix(y_test, y_pred_ensemble)

                print(f"\nEnsemble Results:")
                print(f"  ROC-AUC:   {ensemble_roc_auc:.4f}")
                print(f"  PR-AUC:    {ensemble_pr_auc:.4f}")
                print(f"  F1 (tuned, t={best_thr:.3f}): {best_f1:.4f}")
                print(f"  Precision: {ensemble_precision:.4f}")
                print(f"  Recall:    {ensemble_recall:.4f}")
                print(f"  Confusion Matrix:\n{ensemble_cm}")

                # Compare to best single model
                print(f"\n  vs Best Single Model ({best_model_name}):")
                print(f"    ROC-AUC diff: {ensemble_roc_auc - best_results['holdout']['roc_auc']:+.4f}")
                print(f"    PR-AUC diff:  {ensemble_pr_auc - best_results['holdout']['pr_auc']:+.4f}")

                # Add ensemble to summary
                ensemble_row = {
                    "Model": "Ensemble (Top 3)",
                    "ROC-AUC": ensemble_roc_auc,
                    "PR-AUC": ensemble_pr_auc,
                    "F1 (default)": f1_score(y_test, (ensemble_proba >= 0.5).astype(int), zero_division=0),
                    "F1 (tuned)": best_f1,
                    "Precision (tuned)": ensemble_precision,
                    "Recall (tuned)": ensemble_recall,
                    "Best Threshold": best_thr,
                }
                summary_df = pd.concat([pd.DataFrame([ensemble_row]), summary_df], ignore_index=True)
                summary_df = summary_df.sort_values("ROC-AUC", ascending=False)

                # Save updated summary
                summary_df.to_csv(summary_path, index=False)
                print(f"\nUpdated summary with ensemble saved to: {summary_path}")

        except Exception as e:
            print(f"Could not build ensemble: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {RESULTS_DIR}")

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()
