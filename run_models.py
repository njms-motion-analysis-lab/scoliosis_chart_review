from chart_review_scoliosis_time_predictor import ScoliosisTimePredictor
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from catboost import CatBoostClassifier

# Configure classification models with proper balancing
CLASSIFICATION_MODELS = {
    "RandomForestClassifier": {
        "classifier": RandomForestClassifier(class_weight='balanced'),
        "param_grid": {
            "classifier__n_estimators": [50, 150],
            "classifier__max_depth": [1, 3, 5, 7],
            "classifier__min_samples_split": [2, 3],
            "classifier__min_samples_leaf": [1, 2, 3],
            "classifier__max_features": ["sqrt", "log2"]
        }
    },
    # Uncomment or add additional models as needed:
    # "LogisticRegression": {
    #     "classifier": LogisticRegression(class_weight='balanced', max_iter=1000),
    #     "param_grid": {
    #         "classifier__C": [0.01, 0.1, 1, 10],
    #         "classifier__penalty": ["l1", "l2"],
    #         "classifier__solver": ["liblinear"]
    #     }
    # },
    # "XGBClassifier": {
    #     "classifier": XGBClassifier(random_state=42, eval_metric='logloss'),
    #     "param_grid": {
    #         "classifier__n_estimators": [100, 200],
    #         "classifier__max_depth": [3, 6],
    #         "classifier__learning_rate": [0.01, 0.1],
    #         "classifier__scale_pos_weight": [1, 5, 10]
    #     }
    # },
    "CatBoostClassifier": {
        "classifier": CatBoostClassifier(verbose=False, random_state=42),
        "param_grid": {
            "classifier__iterations": [100, 200],
            "classifier__depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1]
        }
    }
}
RAW_DATA_FOLDER = "raw_chart_data"  # Updated location
RESULTS_FOLDER = "results"          # Output folder
TARGET = "tothlos"                  # Target column

os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Process each raw data file.
for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
    for filename in filenames:
        csv_path = os.path.join(subdir, filename)
        stp = ScoliosisTimePredictor(csv_path=csv_path)
        
        print("Processing file:", filename)
        
        # Training DataFrame CSV path.
        training_df_csv = os.path.join(RESULTS_FOLDER, f"training_df_{TARGET}.csv")
        
        if os.path.exists(training_df_csv):
            print(f"Loading existing training DataFrame from {training_df_csv}")
            df = pd.read_csv(training_df_csv)
        else:
            print("Generating training DataFrame...")
            df = stp.generate_training_dataframe(target_col=TARGET)
            print(f"Saving training DataFrame to {training_df_csv}")
            df.to_csv(training_df_csv, index=False)
        
        # Run grid search pipeline.
        best_pipeline, best_metrics, best_model_name, X_test = stp.grid_search_pipeline(
            data=df, 
            target_column=TARGET,
            models=CLASSIFICATION_MODELS,
            bin_string="x > 5"
        )
        print("Best model:", best_model_name)
        
        # Compute SHAP values for the best model.
        shap_data = stp.compute_shap_values(best_pipeline, X_test, display_plot=False)
        
        # Save performance and SHAP summary results to CSV.
        stp.save_results_to_csv(best_model_name, best_metrics, shap_data, TARGET, RESULTS_FOLDER)