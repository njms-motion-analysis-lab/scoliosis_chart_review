from chart_review_scoliosis_time_predictor import ScoliosisTimePredictor
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os
from datetime import datetime

from catboost import CatBoostClassifier


# ============================================================================
# Configuration
# ============================================================================

RAW_DATA_FOLDER = "raw_chart_data"  # Input data location
RESULTS_FOLDER = "results"          # Output folder for results
TARGET = "reoperation"              # Target column to predict

# ============================================================================
# Debug & Performance Configuration
# ============================================================================
# Set DEBUG_MODE = True for faster iteration during development
# Reduces CV folds, uses simpler parameter grids, smaller models
DEBUG_MODE = True

# Memory control for parallel jobs
# Set N_JOBS = 2 when running via Claude Code to prevent RAM exhaustion
# Set N_JOBS = -1 (all cores) when running manually in terminal
N_JOBS = 8  # Safe default; change to -1 for max speed when running manually

# Feature Selection Configuration
FILTER_FEATURES = True              # Enable feature importance filtering
TOP_N_FEATURES = None               # Keep only top N features (None = use threshold)
MIN_FEATURE_IMPORTANCE = 0.0001      # Remove features below this importance

# Sampling Strategy for Class Imbalance
# Options: "smote", "borderline", "adasyn"
# Note: Regular SMOTE works well for this dataset based on testing
SAMPLING_STRATEGY = "smote"         # Regular SMOTE - tested to work well for ~2% positive rate
USE_SMOTE = True                    # Enable oversampling

# Cross-Validation Strategy - DEBUG_MODE uses faster settings
# "stratified" = simple 5-fold, "repeated" = 5-fold x 3 repeats (more robust but slower)
CV_STRATEGY = "stratified" if DEBUG_MODE else "repeated"
CV_N_SPLITS = 3 if DEBUG_MODE else 5
CV_N_REPEATS = 1 if DEBUG_MODE else 3  # Only used if CV_STRATEGY = "repeated"


# ============================================================================
# Active Classification Models
# ============================================================================

# ============================================================================
# Debug Mode Parameter Grids (minimal for fast iteration)
# ============================================================================
DEBUG_CLASSIFICATION_MODELS = {
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

# ============================================================================
# Full Parameter Grids (for production runs)
# ============================================================================
FULL_CLASSIFICATION_MODELS = {
    # CatBoostClassifier - BEST PERFORMER (ROC-AUC: 0.839, F1: 0.653)
    "CatBoostClassifier": {
        "classifier": CatBoostClassifier(verbose=False, random_state=42),
        "param_grid": {
            "classifier__auto_class_weights": ['Balanced'],
            "classifier__iterations": [500, 700],
            "classifier__depth": [5, 6, 7],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__l2_leaf_reg": [3, 5],
            "classifier__min_data_in_leaf": [5, 10],
        }
    },
    # XGBClassifier - Strong performer (ROC-AUC: 0.825, F1: 0.652)
    "XGBClassifier": {
        "classifier": XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=N_JOBS),
        "param_grid": {
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth": [5, 6, 7],
            "classifier__learning_rate": [0.01, 0.03, 0.1],
            "classifier__scale_pos_weight": [3, 5, 10],
            "classifier__min_child_weight": [3, 5],
            "classifier__subsample": [0.8],
            "classifier__colsample_bytree": [0.8],
            "classifier__reg_alpha": [0, 0.1],
            "classifier__reg_lambda": [1, 3],
        }
    },
    # RandomForestClassifier - ROC-AUC: 0.785
    "RandomForestClassifier": {
        "classifier": RandomForestClassifier(n_jobs=N_JOBS, random_state=42),
        "param_grid": {
            "classifier__n_estimators": [150, 200, 300],
            "classifier__max_depth": [8, 12, 16],
            "classifier__min_samples_split": [5, 10],
            "classifier__min_samples_leaf": [2, 4],
            "classifier__max_features": ["sqrt"],
            "classifier__class_weight": ["balanced_subsample"],
        }
    },
    # LogisticRegression - Simple baseline
    "LogisticRegression": {
        "classifier": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "classifier__C": [0.1],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["liblinear"],
            "classifier__class_weight": ["balanced"]
        }
    },
}

# Select models based on DEBUG_MODE
CLASSIFICATION_MODELS = DEBUG_CLASSIFICATION_MODELS if DEBUG_MODE else FULL_CLASSIFICATION_MODELS


# ============================================================================
# Inactive Models (uncomment to enable)
# ============================================================================

# CLASSIFICATION_MODELS = {

    # "RandomForestRegressor": {
    #     "classifier": RandomForestRegressor(random_state=42, n_jobs=-1),
    #     "param_grid": {
    #         "classifier__n_estimators": [200, 600],
    #         "classifier__max_depth": [None, 8, 16],
    #         "classifier__min_samples_split": [2, 5, 10],
    #         "classifier__min_samples_leaf": [1, 2, 4],
    #         "classifier__max_features": ["sqrt", "log2", 0.5],
    #     },
    # },


# }


# ============================================================================
# Helper Functions
# ============================================================================

def get_result_paths(results_folder, data_file_name, model_name):
    """
    Build the folder structure for organized results.

    Structure:
        results/[data_file]/[model]/heatmaps/
        results/[data_file]/[model]/shap_summaries/
        results/[data_file]/[model]/performance_summaries/

    Args:
        results_folder: Base results directory
        data_file_name: Name of the raw data CSV file (without .csv)
        model_name: Name of the ML model

    Returns:
        dict with paths for 'heatmaps', 'shap_summaries', 'performance_summaries', and 'base'
    """
    base_path = os.path.join(results_folder, data_file_name, model_name)
    paths = {
        "base": base_path,
        "heatmaps": os.path.join(base_path, "heatmaps"),
        "shap_summaries": os.path.join(base_path, "shap_summaries"),
        "performance_summaries": os.path.join(base_path, "performance_summaries"),
    }

    # Create all directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def load_or_generate_training_df(predictor, filename, target, results_folder):
    """
    Load cached training DataFrame if it exists, otherwise generate and save it.

    Args:
        predictor: ScoliosisTimePredictor instance
        filename: Name of the CSV file being processed (e.g., "data.csv")
        target: Target column name
        results_folder: Directory to save/load training DataFrames

    Returns:
        pandas.DataFrame: Training DataFrame ready for modeling
    """
    # Remove .csv extension for naming
    base_name = filename[:-4] if filename.endswith('.csv') else filename

    # Save training_df in the data file's subfolder (not model-specific)
    data_folder = os.path.join(results_folder, base_name)
    os.makedirs(data_folder, exist_ok=True)
    training_df_csv = os.path.join(data_folder, f"training_df_{target}_{base_name}.csv")

    print("Generating training DataFrame...")
    df = predictor.generate_training_dataframe(target_col=target)
    print(f"Saving training DataFrame to {training_df_csv}")
    df.to_csv(training_df_csv, index=False)

    return df


def process_csv_file(filename, csv_path, target, results_folder, classification_models):
    """
    Process a single CSV file: load data, train ALL models, compute SHAP, and save results for each.

    Args:
        filename: Name of the CSV file (e.g., "data.csv")
        csv_path: Full path to the CSV file
        target: Target column to predict
        results_folder: Directory to save results
        classification_models: Dictionary of model configurations
    """
    print(f"Processing file: {filename}")

    # Initialize predictor for this dataset
    predictor = ScoliosisTimePredictor(csv_path=csv_path)

    # Load or generate training DataFrame
    df = load_or_generate_training_df(predictor, filename, target, results_folder)

    # Set up cross-validation strategy
    if CV_STRATEGY == "repeated":
        cv_strategy = RepeatedStratifiedKFold(
            n_splits=CV_N_SPLITS,
            n_repeats=CV_N_REPEATS,
            random_state=42
        )
        print(f"Using RepeatedStratifiedKFold: {CV_N_SPLITS} splits x {CV_N_REPEATS} repeats")
    else:
        cv_strategy = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=42)
        print(f"Using StratifiedKFold: {CV_N_SPLITS} splits")

    # Run grid search for ALL models (return_all_models=True)
    all_results, best_model_name, X_test, y_test = predictor.grid_search_pipeline(
        data=df,
        target_column=target,
        models=classification_models,
        bin_string=None,
        return_all_models=True,
        use_smote=USE_SMOTE,
        sampling_strategy=SAMPLING_STRATEGY,
        filter_features=FILTER_FEATURES,
        top_n_features=TOP_N_FEATURES,
        min_feature_importance=MIN_FEATURE_IMPORTANCE,
        cv_strategy=cv_strategy,
        n_jobs=N_JOBS
    )
    print(f"Best model: {best_model_name}")

    base_name = filename[:-4] if filename.endswith('.csv') else filename

    # Save results for ALL models, not just the best
    for model_name, result in all_results.items():
        print(f"\n=== Saving results for {model_name} ===")
        pipeline = result["estimator"]
        metrics = result["metrics"]

        # Get organized result paths for this model
        result_paths = get_result_paths(results_folder, base_name, model_name)

        # Compute SHAP values for this model
        try:
            shap_data = predictor.compute_shap_values(pipeline, X_test)

            # Save performance metrics and SHAP summary to CSV
            predictor.save_results_to_csv(metrics, shap_data, target, result_paths)

            # Generate and save SHAP visualization for all models
            shap_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predictor.show_shap_beeswarm(
                shap_data,
                X_test,
                show_plot=False,
                save_plot=True,
                plot_filename=os.path.join(result_paths["heatmaps"], f"{target}_{shap_timestamp}.png")
            )
        except Exception as e:
            print(f"Error computing SHAP for {model_name}: {e}")
            # Still save performance metrics even if SHAP fails (use same flat format)
            err_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            perf_csv = os.path.join(result_paths["performance_summaries"], f"{target}_{err_timestamp}.csv")

            # Flatten metrics for easy Excel viewing (same logic as save_results_to_csv)
            holdout = metrics.get("holdout", {})
            tuned = metrics.get("threshold_tuned", {})
            cm = metrics.get("confusion_matrix")
            tn, fp, fn, tp = 0, 0, 0, 0
            if cm is not None:
                try:
                    tn, fp = cm[0][0], cm[0][1]
                    fn, tp = cm[1][0], cm[1][1]
                except (IndexError, TypeError):
                    pass
            flat_metrics = {
                "accuracy": holdout.get("accuracy"),
                "f1": holdout.get("f1"),
                "auc": holdout.get("roc_auc"),
                "precision": holdout.get("precision"),
                "recall": holdout.get("recall"),
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "tuned_threshold": tuned.get("thr") if tuned else None,
                "tuned_f1": tuned.get("f1") if tuned else None,
                "tuned_accuracy": tuned.get("acc") if tuned else None,
                "tuned_precision": tuned.get("prec") if tuned else None,
                "tuned_recall": tuned.get("rec") if tuned else None,
                "best_params": str(metrics.get("best_params", {})),
            }
            perf_df = pd.DataFrame([flat_metrics])
            perf_df.to_csv(perf_csv, index=False)
            print(f"Saved performance metrics (without SHAP) to {perf_csv}")

    # =========================================================================
    # Performance Curves and Model Comparison (like claude_run_code.py)
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_folder = os.path.join(results_folder, base_name)

    # --- ROC Curves for all models ---
    try:
        plt.figure(figsize=(10, 8))

        for model_name, result in all_results.items():
            metrics = result["metrics"]
            y_proba = metrics.get("y_proba")
            y_test_binned = metrics.get("y_test")

            if y_proba is not None and y_test_binned is not None:
                fpr, tpr, _ = roc_curve(y_test_binned, y_proba)
                roc_auc = metrics["holdout"].get("roc_auc", 0)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {base_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        roc_path = os.path.join(data_folder, f"roc_curves_{target}_{timestamp}.png")
        plt.savefig(roc_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"\nROC curves saved to: {roc_path}")
    except Exception as e:
        print(f"Could not save ROC curves: {e}")

    # --- Precision-Recall Curves for all models ---
    try:
        plt.figure(figsize=(10, 8))

        for model_name, result in all_results.items():
            metrics = result["metrics"]
            y_proba = metrics.get("y_proba")
            y_test_binned = metrics.get("y_test")

            if y_proba is not None and y_test_binned is not None:
                precision_vals, recall_vals, _ = precision_recall_curve(y_test_binned, y_proba)
                pr_auc = metrics["holdout"].get("pr_auc", 0)
                plt.plot(recall_vals, precision_vals, label=f"{model_name} (PR-AUC={pr_auc:.3f})")

        # Baseline (proportion of positives)
        if y_test_binned is not None:
            baseline = y_test_binned.mean()
            plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {base_name}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        pr_path = os.path.join(data_folder, f"pr_curves_{target}_{timestamp}.png")
        plt.savefig(pr_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Precision-Recall curves saved to: {pr_path}")
    except Exception as e:
        print(f"Could not save PR curves: {e}")

    # --- Model Comparison Summary CSV ---
    try:
        summary_data = []
        for model_name, result in all_results.items():
            metrics = result["metrics"]
            holdout = metrics.get("holdout", {})
            tuned = metrics.get("threshold_tuned") or {}
            summary_data.append({
                "Model": model_name,
                "ROC-AUC": holdout.get("roc_auc"),
                "PR-AUC": holdout.get("pr_auc"),
                "F1 (default)": holdout.get("f1"),
                "F1 (tuned)": tuned.get("f1"),
                "Precision (default)": holdout.get("precision"),
                "Precision (tuned)": tuned.get("prec"),
                "Recall (default)": holdout.get("recall"),
                "Recall (tuned)": tuned.get("rec"),
                "Accuracy": holdout.get("accuracy"),
                "Best Threshold": tuned.get("thr"),
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values("ROC-AUC", ascending=False)

        summary_path = os.path.join(data_folder, f"model_comparison_{target}_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Model comparison summary saved to: {summary_path}")

        # Print summary to console
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
    except Exception as e:
        print(f"Could not save model comparison summary: {e}")


# ============================================================================
# Main Orchestration
# ============================================================================

def main():
    """
    Main function: traverse raw data folder and process each CSV file.
    """
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Process each CSV file in the raw data folder
    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        # Files are processed alphabetically
        for filename in filenames:
            # Skip hidden files and non-CSV files
            if filename.startswith('.') or not filename.endswith('.csv'):
                print(f"Skipping {filename}")
                continue

            csv_path = os.path.join(subdir, filename)

            # Process file with error handling
            try:
                process_csv_file(filename, csv_path, TARGET, RESULTS_FOLDER, CLASSIFICATION_MODELS)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                print("Continuing to next file...")
                continue


if __name__ == "__main__":
    # Uncomment the line below and move it around to debug
    # import pdb; pdb.set_trace()
    main()
