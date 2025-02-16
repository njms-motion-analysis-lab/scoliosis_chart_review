import os
import re
import pandas as pd
from datetime import datetime
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy  # new import for feature selection
from sklearn.ensemble import RandomForestClassifier

# Global constants used in feature engineering
KNOWN_COMPOSITES = {
    "any_ssi": ["dsupinfec", "wndinfd", "orgspcssi", "dorgspcssi"],
    "any_reop": ["reoperation", "retorrelated", "reoperation2"]  # example
}

POTENTIAL_BINARY = [
    "sex", "ethnicity_hispanic", "transfus", "inout", "dnr", "prem_birth",
    "ventilat", "asthma", "oxygen_sup", "tracheostomy", "stillinhosp",
    "death30yn", "oxygen_at_discharge", "malignancy", "nutr_support",
    "prsepis", "inotr_support", "cpr_prior_surg", "preop_covid", "postop_covid",
    "ped_sap_infection", "ped_sap_prophylaxis", "ped_sap_redosed",
    "ped_spn_antibio_wnd", "ped_spn_antifib", "ped_spn_trnsvol_cell",
    "ped_spn_trnsvol_allogen", "ped_spn_post_trnsvol_cell",
    "ped_spn_post_trnsvol_allogen"
]


class ScoliosisTimePredictor:
    def __init__(self, csv_path="raw_data/v3filteredaiscases.csv"):
        """
        Initializes the ScoliosisTimePredictor.
        Only csv_path is stored since only a few methods are used.
        """
        self.csv_path = csv_path

    def prepare_train_test(self, data, target_column, test_size=0.2, random_state=42, bin_string=None):
        """
        Splits the data into training and testing sets.
        If bin_string is provided, converts the target to binary using the condition.
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        if bin_string is not None:
            y = self.convert_to_binary(data=X, target=y, statement=bin_string)

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def grid_search_pipeline(
        self, 
        data, 
        target_column="tothlos",  # Change as needed.
        test_size=0.2, 
        random_state=42, 
        models=None,
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        bin_string=None,
        use_boruta=False
    ):
        """
        Splits data, optionally applies Boruta feature selection, performs grid searches
        for all models, and returns the best estimator, its performance metrics, model name,
        and the test set used for evaluation.
        """
        print("=== Starting grid search pipeline ===")
        if models is None or not isinstance(models, dict):
            raise ValueError("You must provide a valid dictionary of models and param grids.")

        print("Preparing training and testing sets...")
        X_train, X_test, y_train, y_test = self.prepare_train_test(
            data, target_column, test_size, random_state, bin_string=bin_string
        )
        
        if use_boruta:
            X_train, X_test = self._apply_boruta(X_train, y_train, X_test, random_state)
        else:
            print("Skipping Boruta feature selection (use_boruta is False).")
        
        best_estimator = None
        best_score = -np.inf
        best_model_name = None
        best_metrics = {}

        for model_name, model_info in models.items():
            print(f"\n--- Starting grid search for {model_name} ---")
            estimator, metrics = self._run_grid_search(
                model_name, model_info, X_train, y_train, X_test, y_test, cv_strategy
            )
            if metrics["auc"] > best_score:
                best_score = metrics["auc"]
                best_estimator = estimator
                best_model_name = model_name
                best_metrics = metrics

        print(f"\n=== Best Overall Model: {best_model_name} ===")
        print(f"Final AUC: {best_metrics['auc']:.3f}")
        print("Final Confusion Matrix:")
        print(best_metrics["confusion_matrix"])
        
        # Return also X_test for subsequent SHAP analysis.
        return best_estimator, best_metrics, best_model_name, X_test

    def _apply_boruta(self, X_train, y_train, X_test, random_state):
        """
        Applies Boruta feature selection on the training set and filters X_test accordingly.
        """
        print(">>> Running Boruta feature selection on training data...")
        from boruta import BorutaPy  # Local import
        boruta_estimator = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        boruta_selector = BorutaPy(
            estimator=boruta_estimator,
            n_estimators='auto',
            verbose=2,
            random_state=random_state,
            max_iter=100,      # adjust if needed
            two_step=True,     # less strict, optional
            alpha=0.1          # significance level; increase to be less strict
        )
        boruta_selector.fit(X_train.values, y_train.values)
        selected_features = X_train.columns[boruta_selector.support_].tolist()
        print(">>> Boruta selected features:", selected_features)
        
        if selected_features:
            print(">>> Filtering training and testing sets to selected features.")
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
        else:
            print(">>> Warning: Boruta did not select any features, proceeding with all features.")
        
        return X_train, X_test

    def _run_grid_search(self, model_name, model_info, X_train, y_train, X_test, y_test, cv_strategy):
        """
        Runs grid search for a given model and returns the best estimator and its performance metrics.
        """
        classifier = model_info["classifier"]
        param_grid = model_info["param_grid"]
        pipeline = Pipeline([("classifier", classifier)])
        
        scoring = {
            'auc': 'roc_auc',
            'accuracy': 'accuracy',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit='auc',
            cv=cv_strategy,
            n_jobs=-1,
            verbose=1
        )
        
        print(f">>> Fitting grid search for {model_name}...")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f">>> Completed grid search for {model_name}.")
        print(f">>> Best parameters for {model_name}: {grid_search.best_params_}")
        
        # Evaluate on the test set.
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'best_params': grid_search.best_params_,
            'feature_importances': (
                best_model.named_steps['classifier'].feature_importances_
                if hasattr(best_model.named_steps['classifier'], 'feature_importances_')
                else None
            )
        }
        
        print(f">>> Metrics for {model_name}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}")
        return best_model, metrics

    def compute_shap_values(self, best_estimator, X_test, display_plot=False):
        """
        Computes SHAP values for the best estimator and returns a dictionary containing:
          - raw shap values (array)
          - a summary DataFrame of mean absolute SHAP values per feature.
        Optionally displays a SHAP summary plot.
        """
        print(">>> Computing SHAP values...")
        best_classifier = best_estimator.named_steps['classifier']
        explainer = shap.TreeExplainer(best_classifier)
        raw_shap_values = explainer.shap_values(X_test)
        
        # Handle cases with multiple output dimensions.
        raw_shape = np.array(raw_shap_values).shape
        if len(raw_shape) == 3 and raw_shape[2] == 2:
            shap_values = raw_shap_values[:, :, 1]
        else:
            shap_values = raw_shap_values
        
        # Compute mean absolute SHAP value per feature.
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_summary_df = pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False)
        
        if display_plot:
            print(">>> Displaying SHAP summary plot...")
            shap.summary_plot(shap_values, X_test, max_display=25, plot_type="dot")
        
        print(">>> SHAP computation complete.")
        return {"shap_values": shap_values, "shap_summary": shap_summary_df}

    def save_results_to_csv(self, best_model_name, best_metrics, shap_data, target, results_folder):
        """
        Saves performance metrics and SHAP summary to CSV files.
        Filenames include the target and a timestamp for easy tracking.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        performance_csv = os.path.join(results_folder, f"performance_{best_model_name}_{target}_{timestamp}.csv")
        shap_csv = os.path.join(results_folder, f"shap_summary_{best_model_name}_{target}_{timestamp}.csv")
        
        # Save performance metrics.
        perf_df = pd.DataFrame([best_metrics])
        perf_df.to_csv(performance_csv, index=False)
        print(f"Saved performance metrics to {performance_csv}")
        
        # Save SHAP summary.
        shap_data["shap_summary"].to_csv(shap_csv, index=False)
        print(f"Saved SHAP summary to {shap_csv}")

    def run_random_forest_regression_with_shap(self, X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
        """
        Trains a Random Forest regressor, prints evaluation metrics, and displays a SHAP beeswarm plot.
        """
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Random Forest Regression Results:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")

        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        print(f"SHAP values shape: {shap_values.shape}")
        print("Displaying SHAP beeswarm plot...")
        shap.summary_plot(shap_values, X_test, max_display=40, plot_type="dot")

        return rf, {"MAE": mae, "MSE": mse, "R2": r2}

    def random_forest_pipeline_with_shap(self, data, target_column="tothlos", test_size=0.2, random_state=42):
        """
        Full pipeline to train and evaluate a Random Forest regression model with SHAP visualization.
        Adjusts the target for binary cases if needed.
        """
        print(f"Preparing pipeline for target: {target_column}")
        data = self.adjust_for_binary_targets(data, target_column)
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = self.prepare_train_test(data, target_column, test_size, random_state)
        print("Training Random Forest model with SHAP visualization...")
        return self.run_random_forest_regression_with_shap(X_train, X_test, y_train, y_test)

    def refine_dataframe(self, df, target_column, max_iter=500, random_state=42, two_step=True, alpha=0.1):
        """
        Refines the given DataFrame by selecting the best features using Boruta.
        
        If the target is binary (2 unique values), it uses a RandomForestClassifier;
        otherwise, it uses a RandomForestRegressor.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.
            max_iter (int): Maximum number of iterations for Boruta (default: 500).
            random_state (int): Random state for reproducibility.
            two_step (bool): Whether to use the two-step procedure (can be less strict).
            alpha (float): Significance level for feature acceptance (increase to be less strict).
            
        Returns:
            pd.DataFrame: A refined DataFrame containing only the selected features 
                        along with the target column.
        """
        # Separate features and target
    
    

        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print("Target unique values:", y.unique())
        print("Number of unique target values:", y.nunique())
        
        # Choose estimator based on target type.
        # Since your target gets converted to binary, we use RandomForestClassifier.
        if y.nunique() == 2:
            print("Using RandomForestClassifier for Boruta feature selection.")
            estimator = RandomForestClassifier(n_estimators=1000, random_state=random_state, n_jobs=-1)
        else:
            print("Using RandomForestRegressor for Boruta feature selection.")
            estimator = RandomForestRegressor(n_estimators=1000, random_state=random_state, n_jobs=-1)
        
        # Initialize and run Boruta with adjustable parameters.
        boruta_selector = BorutaPy(
            estimator,
            n_estimators='auto',
            verbose=2,
            random_state=random_state,
            max_iter=max_iter,
            two_step=two_step,
            alpha=alpha
        )
        boruta_selector.fit(X.values, y.values)
        
        # Retrieve the selected features
        selected_features = X.columns[boruta_selector.support_].tolist()
        print("Boruta selected features:", selected_features)
        
        # If no features are selected, warn and return the original DataFrame.
        if not selected_features:
            print("Warning: Boruta did not select any features. Returning the original DataFrame.")
            return df
        
        # Return a DataFrame with the selected features plus the target column.
        refined_df = df[selected_features + [target_column]]
        return refined_df

    def load_and_clean_data(self):
        """
        Loads and cleans the scoliosis data from the CSV path.
        """
        try:
            data = pd.read_csv(self.csv_path)
            data.drop_duplicates(inplace=True)
            data.fillna(method="ffill", inplace=True)
            data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
            for date_col in ["date_of_surgery", "date_of_birth"]:
                if date_col in data.columns:
                    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_path}")
            return pd.DataFrame()

    def adjust_for_binary_targets(self, data, target_column):
        """
        If the target column is binary (two unique values), converts it to integer (0/1).
        """
        if data[target_column].nunique() == 2:
            print(f"Binary target detected for '{target_column}'. Encoding as 0/1.")
            data[target_column] = data[target_column].astype(int)
        return data

    def convert_to_binary(self, data, target, statement):
        """
        Converts a numeric column or Series to binary (0/1) based on the condition provided in 'statement'.
        For example, 'x >= 5' converts values >= 5 to 1 and the rest to 0.
        """
        
        if isinstance(target, str):
            series = data[target]
        elif isinstance(target, pd.Series):
            series = target
        else:
            raise ValueError("target must be either a column name (str) or a pandas Series.")

        try:
            condition = eval(statement, {"x": series, "np": np, "pd": pd})
        except Exception as e:
            raise ValueError(f"Error evaluating the statement '{statement}': {e}")

        binary_series = pd.Series(np.where(condition, 1, 0), index=series.index)
        return binary_series

    def generate_training_dataframe(self, target_col="any_ssi"):
        """
        Loads, cleans, and feature-engineers the data to create a training DataFrame.
        """
        data = self.load_and_clean_data()
        if data.empty:
            return pd.DataFrame()

        data.columns = data.columns.str.lower()

        # Create composite target if needed and drop correlated columns.
        data = ScoliosisFeatureEngineeringService.generate_comp_target(data, target_col=target_col)
        data = ScoliosisFeatureEngineeringService.drop_perfectly_correlated_columns(data, target_column=target_col)

        def replace_with_nan(data, missing_tokens=None):
            if missing_tokens is None:
                missing_tokens = ["-99", "null", "#null!", "na", "n/a", "", " "]
            for col in data.columns:
                data[col] = data[col].astype(str).str.strip()
            for token in missing_tokens:
                pattern = rf"(?i)^{re.escape(token)}$"
                data.replace(to_replace=pattern, value=np.nan, regex=True, inplace=True)
            return data

        columns_to_drop_if_empty = ["ped_sap_name1"]
        data = ScoliosisFeatureEngineeringService.remove_empty_rows(data, columns_to_drop_if_empty)
        data = replace_with_nan(data, missing_tokens=["-99", "null", "#null!", "na", "n/a", "", " "])
        data = ScoliosisFeatureEngineeringService.encode_binary(data, POTENTIAL_BINARY)
        cat_cols = ScoliosisFeatureEngineeringService.find_potential_string_categorical_cols(data)
        data = ScoliosisFeatureEngineeringService.encode_categorical(data, cat_cols)

        # Antibiotic regimen logic.
        data = ScoliosisFeatureEngineeringService.multi_hot_encode_abx(data)
        data = ScoliosisFeatureEngineeringService.encode_abx_combinations(data)
        data = ScoliosisFeatureEngineeringService.encode_bmi(data)
        data = ScoliosisFeatureEngineeringService.drop_low_variance_columns(data)

        # Clip numeric values to float32 limits.
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].clip(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max
        )

        data = ScoliosisFeatureEngineeringService.make_rf_compatible(data)
        data = ScoliosisFeatureEngineeringService.rename_for_xgb_compatibility(data)

        return data


class ScoliosisFeatureEngineeringService:
    @staticmethod
    def generate_comp_target(data, target_col, axis=1):
        """
        If target_col is not present but is a known composite, creates it
        by OR’ing binary versions of the composite columns.
        """
        def standardize_ssi_column(value):
            val_str = str(value).strip().lower()
            if val_str in ["no complication", "-99", "nan", "", "null", "none"]:
                return 0
            else:
                return 1

        if target_col not in data.columns and target_col in KNOWN_COMPOSITES:
            composite_cols = KNOWN_COMPOSITES[target_col]
            missing = [c for c in composite_cols if c not in data.columns]
            if missing:
                print(f"Warning: Composite target '{target_col}' missing columns: {missing}.")
            data[target_col] = 0
            for c in composite_cols:
                if c in data.columns:
                    data[c] = data[c].apply(standardize_ssi_column).astype(int)
                    data[target_col] = data[target_col] | data[c]
            print(f"Created composite column '{target_col}' from {composite_cols}.")

        if target_col not in data.columns:
            print(f"Target column '{target_col}' not found in data. Returning empty DataFrame.")
            return pd.DataFrame()
        return data

    @staticmethod
    def multi_hot_encode_abx(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates multi-hot (binary) columns for each antibiotic from columns starting with "ped_sap_name".
        """
        ABX_NAME_MAP = {
            'ampicillin without sulbactam': 'ampicillin',
            'ampicillin': 'ampicillin',
            'cefazolin': 'cefazolin',
            'cefoxitin': 'cefoxitin',
            'ceftazidime': 'ceftazidime',
            'ceftriaxone': 'ceftriaxone',
            'vancomycin': 'vancomycin',
            'gentamicin': 'gentamicin',
            'clindamycin': 'clindamycin',
            'metronidazole': 'metronidazole',
            'doxycycline': 'doxycycline',
        }
        unique_abx = set(ABX_NAME_MAP.values())
        for abx in unique_abx:
            df[f"used_{abx}"] = False

        for col in df.columns:
            if not col.lower().startswith("ped_sap_name"):
                continue
            raw_abx_name = re.sub(r"^ped_sap_name\d+_", "", col, flags=re.IGNORECASE).lower()
            for pattern, canonical in ABX_NAME_MAP.items():
                if pattern in raw_abx_name:
                    df.loc[df[col] == True, f"used_{canonical}"] = True
                    break
        return df

    @staticmethod
    def encode_abx_combinations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates row by row over the multi-hot encoded antibiotic columns and creates new columns
        for each unique combination.
        """
        abx_cols = [col for col in df.columns if col.startswith("used_")]
        combo_map = {}
        for idx, row in df.iterrows():
            used_list = [col.replace("used_", "") for col in abx_cols if row[col] == True or row[col] == 1]
            used_list.sort()
            combo_key = tuple(used_list)
            if combo_key:
                combo_col = "combo_abx_regimen_" + "_".join(combo_key)
            else:
                combo_col = "single_abx_regimen_"
            if combo_key not in combo_map:
                combo_map[combo_key] = combo_col
                df[combo_col] = 0
            df.at[idx, combo_map[combo_key]] = 1
        return df

    @staticmethod
    def encode_bmi(data):
        """
        Encodes BMI using height and weight columns.
        """
        print("\n=== BMI Encoding Process ===")
        required_cols = {'height', 'weight'}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            print(f"🚨 Critical: Missing required columns {missing_cols} - skipping BMI calculation")
            return data

        def safe_convert(col, conversion_factor):
            try:
                series = pd.to_numeric(data[col], errors='coerce')
                non_convertible = series.isna() & data[col].notna()
                if non_convertible.any():
                    invalid_values = data.loc[non_convertible, col].unique()[:5]
                    print(f"  ⚠️ Non-numeric values in {col}: {len(invalid_values)} unique (e.g., {invalid_values})")
                return series * conversion_factor
            except Exception as e:
                print(f"🚨 Critical error converting {col}: {str(e)}")
                return pd.Series(np.nan, index=data.index)

        print("\n1. Unit Conversion:")
        print("  • Converting height (inches → meters)")
        df = data.assign(
            height_m=safe_convert('height', 0.0254),
            weight_kg=safe_convert('weight', 0.453592)
        )
        print("\n2. Data Validation:")
        height_valid = df['height_m'].between(0.3, 2.5)
        weight_valid = df['weight_kg'].between(2, 300)
        df['height_m'] = np.where(height_valid, df['height_m'], np.nan)
        df['weight_kg'] = np.where(weight_valid, df['weight_kg'], np.nan)
        print(f"  • Height valid: {height_valid.sum():,}/{len(df)}")
        print(f"  • Weight valid: {weight_valid.sum():,}/{len(df)}")

        print("\n3. BMI Calculation:")
        with np.errstate(divide='ignore', invalid='ignore'):
            df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)
        valid_bmi = df['bmi'].notna()
        print(f"  • Successful calculations: {valid_bmi.sum():,}/{len(df)}")
        print(f"  • BMI range: {df.bmi.min():.1f}-{df.bmi.max():.1f}")

        print("\n4. Quality Flags:")
        bmi_conditions = [
            (df['bmi'] < 10) | (df['bmi'] > 60),
            (df['bmi'] < 15) | (df['bmi'] > 40),
            (df['bmi'] < 18.5) | (df['bmi'] > 30)
        ]
        bmi_flags = ['invalid_extreme', 'invalid_clinical', 'unusual_clinical']
        df['bmi_quality_flag'] = np.select(
            condlist=bmi_conditions,
            choicelist=bmi_flags,
            default='valid'
        )
        flag_counts = df['bmi_quality_flag'].value_counts()
        print("  • BMI quality distribution:")
        for flag, count in flag_counts.items():
            print(f"    - {flag}: {count:,} ({count/len(df)*100:.1f}%)")

        print("\n5. Final Integration:")
        data['bmi'] = np.where(
            df['bmi_quality_flag'].isin(['valid', 'unusual_clinical']),
            df['bmi'],
            np.nan
        )
        data['bmi_quality_flag'] = df['bmi_quality_flag']
        print(f"  • Null BMI values: {data['bmi'].isna().sum():,}")
        print("✅ BMI encoding complete\n")
        return data

    @staticmethod
    def drop_low_variance_columns(data, threshold=0.001):
        """
        Drops columns with very low variance or only one unique value.
        """
        print("Checking for low-variance columns...")
        if data.columns.duplicated().any():
            print("Warning: Duplicate column names found; renaming them.")
            data.columns = ScoliosisFeatureEngineeringService.make_unique_column_names(data.columns)

        single_unique_cols = [col for col in data.columns if data[col].nunique(dropna=False) == 1]
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        low_var_numeric_cols = []
        for col in numeric_cols:
            col_series = pd.to_numeric(data[col], errors='coerce').dropna()
            col_var = 0.0 if col_series.empty else col_series.var()
            if pd.isna(col_var):
                col_var = 0.0
            if col_var < threshold:
                low_var_numeric_cols.append(col)
        low_variance_cols = list(set(single_unique_cols + low_var_numeric_cols))
        print(f"Low-variance columns to drop: {low_variance_cols}")
        data = data.drop(columns=low_variance_cols, errors="ignore")
        data = data.drop(columns="htooday", errors="ignore")
        return data

    @staticmethod
    def drop_perfectly_correlated_columns(data, target_column):
        """
        Drops columns that are perfectly (or highly) correlated with the target.
        """
        print(f"Checking for columns correlated with {target_column}...")
        numeric_data = data.select_dtypes(include=[np.number])
        if target_column not in numeric_data.columns:
            raise ValueError(f"Target column '{target_column}' must be numeric for correlation analysis.")
        # For this example, we drop a fixed set of known highly correlated columns.
        correlated_columns = [
            "dsupinfec", "wndinfd", "doptodis", "orgspcssi", "dehis", "nwnd",
            "dwndinfd", "dorgspcssi", "ddehis", "ndehis", "norgspcssi",
            "nwndinfd", "noupneumo", "doupneumo", "reopor2cpt1",
            "retor2related", "unplannedreadmission1", "reoporcpt1"
        ]
        print(f"Columns to drop: {correlated_columns}")
        data = data.drop(columns=correlated_columns, errors="ignore")
        return data

    @staticmethod
    def find_potential_string_categorical_cols(df, max_unique=50):
        """
        Identifies columns that are likely categorical based on their type and number of unique values.
        """
        bool_like_values = {
            "yes", "no", "true", "false", "0", "1", "null",
            "none", "", "nan", "-99", "-1", "male", "female"
        }
        text_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        potential_categorical = []
        for col in text_cols:
            unique_vals = df[col].unique()
            unique_non_nan = [str(v).strip() for v in unique_vals if pd.notna(v)]
            if len(unique_vals) < max_unique:
                normalized = set(val.lower() for val in unique_non_nan)
                if not normalized.issubset(bool_like_values):
                    potential_categorical.append(col)
        return potential_categorical

    @staticmethod
    def rename_for_xgb_compatibility(df):
        """
        Renames columns to remove characters that may cause issues with XGBoost.
        """
        df.columns = df.columns.map(str)
        forbidden_pattern = r"[\[\]<>\(\)]"
        df.columns = [re.sub(forbidden_pattern, "_", col) for col in df.columns]
        return df

    @staticmethod
    def encode_binary(data, binary_columns):
        """
        Encodes columns in binary_columns as 0/1 where possible.
        """
        binary_map = {
            "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, "male": 1, "Yes": 1,
            "no": 0, "n": 0, "false": 0, "f": 0, "0": 0, "female": 0, "No": 0,
        }
        allowed_numerics = {0.0, 1.0, -99.0, -1.0}
        for col in binary_columns:
            col_lower = col.lower()
            if col_lower not in data.columns:
                continue
            original_series = data[col_lower].copy()
            distinct_vals = original_series.dropna().unique()
            skip_encoding = False
            for val in distinct_vals:
                str_val = str(val).strip().lower()
                if str_val in binary_map:
                    continue
                else:
                    try:
                        float_val = float(str_val)
                        if float_val not in allowed_numerics:
                            skip_encoding = True
                            break
                    except ValueError:
                        skip_encoding = True
                        break
            if skip_encoding:
                data[col_lower] = original_series
                continue

            def map_to_binary(x):
                s = str(x).strip().lower()
                if s in binary_map:
                    return float(binary_map[s])
                else:
                    try:
                        f_val = float(s)
                        return 0.0 if f_val != 1.0 else 1.0
                    except ValueError:
                        return np.nan

            temp_encoded = original_series.apply(map_to_binary)
            unique_after = temp_encoded.dropna().unique()
            if len(unique_after) <= 2:
                data[col_lower] = temp_encoded
            else:
                data[col_lower] = original_series
        return data

    @staticmethod
    def encode_categorical(data, categorical_columns, max_categories=50, numeric_threshold=0.9):
        """
        One-hot encodes categorical columns with limited unique values.
        Groups rare categories into "Other" if necessary.
        """
        for col in categorical_columns:
            col_lower = col.lower()
            if col_lower in data.columns:
                non_null_values = data[col_lower].dropna()
                try:
                    numeric_values = pd.to_numeric(non_null_values, errors='coerce')
                    numeric_proportion = numeric_values.notna().mean()
                    if numeric_proportion >= numeric_threshold:
                        continue
                    unique_values = data[col_lower].nunique()
                    if unique_values > max_categories:
                        top_categories = data[col_lower].value_counts().nlargest(max_categories).index
                        data[col_lower] = data[col_lower].apply(lambda x: x if x in top_categories else "Other")
                    data = pd.get_dummies(data, columns=[col_lower], drop_first=True)
                except (TypeError, ValueError):
                    unique_values = data[col_lower].nunique()
                    if unique_values > max_categories:
                        top_categories = data[col_lower].value_counts().nlargest(max_categories).index
                        data[col_lower] = data[col_lower].apply(lambda x: x if x in top_categories else "Other")
                    data = pd.get_dummies(data, columns=[col_lower], drop_first=True)
        return data

    @staticmethod
    def make_rf_compatible(data):
        """
        Processes the DataFrame to handle missing values and converts non-numeric columns to numeric.
        """
        missing_cols = data.columns[data.isna().any()].tolist()
        print(f"Columns with missing values: {missing_cols}")
        for col in missing_cols:
            if data[col].dtype in [np.float64, np.int64]:
                data[col].fillna(data[col].mean(), inplace=True)
            elif data[col].dtype == "object" or data[col].dtype.name == "category":
                data[col].fillna("Unknown", inplace=True)
                data[col] = data[col].astype(str)
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            elif data[col].dtype == "bool":
                data[col].fillna(False, inplace=True)
                data[col] = data[col].astype(int)
            else:
                data[col].fillna(0, inplace=True)
        for col in data.select_dtypes(include=["object", "category"]).columns:
            data[col] = data[col].astype(str)
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        if data.isna().any().any():
            print("Warning: NaN values remain after processing.")
        else:
            print("All missing values handled successfully.")
        return data

    @staticmethod
    def remove_empty_rows(data, columns_to_check):
        """
        Removes rows that have empty or whitespace-only strings or NaN in any of the specified columns.
        """
        mask = pd.Series(True, index=data.index)
        for col in columns_to_check:
            if col in data.columns:
                not_nan = ~pd.isna(data[col])
                not_empty = pd.Series(True, index=data.index)
                non_nan_mask = ~pd.isna(data[col])
                if non_nan_mask.any():
                    not_empty[non_nan_mask] = data.loc[non_nan_mask, col].astype(str).str.strip() != ''
                mask = mask & (not_nan & not_empty)
            else:
                print(f"Warning: Column '{col}' does not exist in DataFrame. Skipping.")
        return data[mask]

    @staticmethod
    def make_unique_column_names(columns):
        """
        Given a list of column names, renames duplicates by appending .1, .2, etc.
        """
        new_cols = []
        name_count = {}
        for col in columns:
            if col not in name_count:
                name_count[col] = 0
                new_cols.append(col)
            else:
                name_count[col] += 1
                new_cols.append(f"{col}.{name_count[col]}")
        return new_cols
