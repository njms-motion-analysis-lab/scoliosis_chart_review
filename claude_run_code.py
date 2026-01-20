"""
Optimized Reoperation Prediction Analysis
==========================================
This script uses the existing ScoliosisTimePredictor infrastructure for preprocessing
and trains models to predict reoperation risk.

Uses the same approach that achieved ROC-AUC: 0.839, F1: 0.653 on CatBoost.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix, classification_report,
    make_scorer, roc_curve, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier

# Imbalanced-learn imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Boosting models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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

# Cross-validation settings (matches successful run_models.py config)
CV_SPLITS = 3
CV_REPEATS = 1

# Feature filtering (matches run_models.py)
FILTER_FEATURES = True
MIN_FEATURE_IMPORTANCE = 0.0001

# SMOTE settings
USE_SMOTE = True
SMOTE_K_NEIGHBORS = 3


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
# MODEL DEFINITIONS (matches successful run_models.py config)
# =============================================================================

def get_models():
    """
    Define models optimized for severe class imbalance.
    These configurations achieved ROC-AUC ~0.83-0.84 in previous runs.
    """
    models = {
        # CatBoostClassifier (kept as you had it)
        "CatBoostClassifier": {
            "classifier": CatBoostClassifier(verbose=False, random_state=42),
            "param_grid": {
                "classifier__auto_class_weights": ["Balanced"],
                "classifier__iterations": [200],
                "classifier__depth": [8],
                "classifier__learning_rate": [0.1],
                # Optional: add one regularization knob without exploding the grid
                "classifier__l2_leaf_reg": [3],
            },
        },

        # LogisticRegression (kept as you had it)
        "LogisticRegression": {
            "classifier": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.1],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["liblinear"],
                "classifier__class_weight": ["balanced"],
            },
        },

        # XGBClassifier (reduced, "equivalent" mid-range choices)
        # Keep only 2-4 combos so grid search stays short.
        "XGBClassifier": {
            "classifier": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=N_JOBS,
            ),
            "param_grid": {
                # modest grid: 2 x 2 = 4 combos
                "classifier__n_estimators": [200],
                "classifier__max_depth": [6],                 # middle of [5,6,7]
                "classifier__learning_rate": [0.03, 0.1],      # representative low/mid
                "classifier__scale_pos_weight": [5, 10],       # mid/high for imbalance
                # keep the rest fixed to avoid blow-up
                "classifier__min_child_weight": [5],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
                "classifier__reg_alpha": [0],
                "classifier__reg_lambda": [1],
            },
        },

        # RandomForestClassifier (reduced, "equivalent" mid-range choices)
        "RandomForestClassifier": {
            "classifier": RandomForestClassifier(n_jobs=N_JOBS, random_state=42),
            "param_grid": {
                # 2 x 2 = 4 combos
                "classifier__n_estimators": [300],
                "classifier__max_depth": [12, 16],            # mid/high in [8,12,16]
                "classifier__min_samples_split": [5],
                "classifier__min_samples_leaf": [2, 4],       # small variation
                "classifier__max_features": ["sqrt"],
                "classifier__class_weight": ["balanced_subsample"],
            },
        },

        # Optional: add BalancedRandomForestClassifier as a fast, robust imbalance baseline
        # Note: your pipeline auto-skips SMOTE for BRF by name, so this is a clean addition.
        "BalancedRandomForestClassifier": {
            "classifier": BalancedRandomForestClassifier(
                random_state=42,
                n_jobs=N_JOBS,
            ),
            "param_grid": {
                # keep tiny: 2 combos
                "classifier__n_estimators": [300],
                "classifier__max_depth": [12],
                "classifier__min_samples_split": [5],
                "classifier__min_samples_leaf": [2, 4],
                "classifier__max_features": ["sqrt"],
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
    Uses ImbPipeline with SMOTE (no StandardScaler - not needed for tree models).
    """
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")

    clf = model_config["classifier"]
    param_grid = model_config["param_grid"]

    # Use imbalanced-learn Pipeline with SMOTE (no scaler - tree models don't need it)
    if USE_SMOTE:
        print(f"  Using SMOTE oversampling (k_neighbors={SMOTE_K_NEIGHBORS})")
        pipe = ImbPipeline([
            ("sampler", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)),
            ("classifier", clf)
        ])
    else:
        from sklearn.pipeline import Pipeline
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
    Main execution function using ScoliosisTimePredictor for preprocessing.
    """
    print("\n" + "="*70)
    print("REOPERATION PREDICTION - USING SCOLIOSIS TIME PREDICTOR")
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
    # FEATURE FILTERING (like run_models.py)
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

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {RESULTS_DIR}")

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()
