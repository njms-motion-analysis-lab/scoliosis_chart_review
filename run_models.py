from chart_review_scoliosis_time_predictor import ScoliosisTimePredictor
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import os

from catboost import CatBoostClassifier


# ============================================================================
# Configuration
# ============================================================================

RAW_DATA_FOLDER = "raw_chart_data"  # Input data location
RESULTS_FOLDER = "results"          # Output folder for results
TARGET = "any_ssi"                  # Target column to predict


# ============================================================================
# Active Classification Models
# ============================================================================

CLASSIFICATION_MODELS = {
    "CatBoostClassifier": {
        "classifier": CatBoostClassifier(verbose=False, random_state=42),
        "param_grid": {
            "classifier__auto_class_weights": ['Balanced'],
            "classifier__iterations": [200, 300, 500],
            "classifier__depth": [4, 5, 6],
            "classifier__learning_rate": [0.01, 0.1],
        }
    }
}


# ============================================================================
# Inactive Models (uncomment to enable)
# ============================================================================

# CLASSIFICATION_MODELS = {
#     "RandomForestClassifier": {
#         "classifier": RandomForestClassifier(class_weight='balanced'),
#         "param_grid": {
#             "classifier__n_estimators": [50, 150],
#             "classifier__max_depth": [1, 3, 5, 7],
#             "classifier__min_samples_split": [2, 3],
#             "classifier__min_samples_leaf": [1, 2, 3],
#             "classifier__max_features": ["sqrt", "log2"]
#         }
#     },
#     "RandomForestRegressor": {
#         "classifier": RandomForestRegressor(random_state=42, n_jobs=-1),
#         "param_grid": {
#             "classifier__n_estimators": [200, 600],
#             "classifier__max_depth": [None, 8, 16],
#             "classifier__min_samples_split": [2, 5, 10],
#             "classifier__min_samples_leaf": [1, 2, 4],
#             "classifier__max_features": ["sqrt", "log2", 0.5],
#         },
#     },
#     "LogisticRegression": {
#         "classifier": LogisticRegression(class_weight='balanced', max_iter=1000),
#         "param_grid": {
#             "classifier__C": [0.01, 0.1, 1, 10],
#             "classifier__penalty": ["l1", "l2"],
#             "classifier__solver": ["liblinear"]
#         }
#     },
#     "XGBClassifier": {
#         "classifier": XGBClassifier(random_state=42, eval_metric='logloss'),
#         "param_grid": {
#             "classifier__n_estimators": [100, 200],
#             "classifier__max_depth": [3, 6],
#             "classifier__learning_rate": [0.01, 0.1],
#             "classifier__scale_pos_weight": [1, 5, 10]
#         }
#     },
# }


# ============================================================================
# Helper Functions
# ============================================================================

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
    training_df_csv = os.path.join(results_folder, f"training_df_{base_name}_{target}.csv")

    if os.path.exists(training_df_csv):
        print(f"Loading existing training DataFrame from {training_df_csv}")
        df = pd.read_csv(training_df_csv)
    else:
        print("Generating training DataFrame...")
        df = predictor.generate_training_dataframe(target_col=target)
        print(f"Saving training DataFrame to {training_df_csv}")
        df.to_csv(training_df_csv, index=False)

    return df


def process_csv_file(filename, csv_path, target, results_folder, classification_models):
    """
    Process a single CSV file: load data, train models, compute SHAP, and save results.

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

    # Run grid search to find best model
    best_pipeline, best_metrics, best_model_name, X_test = predictor.grid_search_pipeline(
        data=df,
        target_column=target,
        models=classification_models,
        bin_string=None
    )
    print(f"Best model: {best_model_name}")

    # Compute SHAP values for model interpretation
    shap_data = predictor.compute_shap_values(best_pipeline, X_test)

    # Save performance metrics and SHAP summary to CSV
    base_name = filename[:-4] if filename.endswith('.csv') else filename
    predictor.save_results_to_csv(base_name, best_model_name, best_metrics, shap_data, target, results_folder)

    # Generate and save SHAP visualization
    predictor.show_shap_beeswarm(
        shap_data,
        X_test,
        show_plot=False,
        save_plot=True,
        plot_filename=f"heatmap_{base_name}_{target}.png"
    )

    # Optional: Additional explanatory plots (currently disabled)
    # prefix = f"{base_name}_{target}_{best_model_name}"
    # predictor.save_explanatory_plots(
    #     shap_data=shap_data,
    #     X_test=X_test,
    #     pipeline=best_pipeline,
    #     out_dir=results_folder,
    #     prefix=prefix,
    #     pairs=[("optime", "anetime")],  # Customize feature pairs as needed
    # )

    # Optional: SHAP dependence plots (currently disabled)
    # predictor.show_shap_dependence_plot(
    #     shap_data, X_test,
    #     feature="anetime",
    #     interaction_feature="optime",
    #     show_plot=True,
    #     save_plot=False
    # )
    # predictor.show_shap_dependence_plot(
    #     shap_data, X_test,
    #     feature="optime",
    #     interaction_feature="anetime",
    #     show_plot=True,
    #     save_plot=False
    # )


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
