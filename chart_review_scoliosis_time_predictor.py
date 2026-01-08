import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    classification_report, make_scorer
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from boruta import BorutaPy
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
    "hxcld", "struct_pulm_ab", "esovar", "prvpcs", "impcogstat", 
    "ped_spn_post_trnsvol_allogen",
    "seizure", "cerebral_palsy", "acq_abnormality", "neuromuscdis", 
    "steroid", "ostomy", "hemodisorder", # ... add any others
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
            estimator, metrics = self._run_grid_search(model_name, model_info, X_train, y_train, X_test, y_test, cv_strategy)
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
        Binary LOS classification: 0 = LOS <= 1 day, 1 = LOS > 1 day.
        Assumes X_* are already preprocessed. Returns metrics with 'auc' key (ROC AUC).
        """

        # ----- 1) Binary binning config -----
        # Two categories: <=1 vs >1 day
        # bin_edges  = model_info.get("bin_edges",  [-1, 1, np.inf])
        # bin_labels = model_info.get("bin_labels", [0, 1])
        # positive_bin = model_info.get("positive_bin", 1)  # 'extended LOS' = >1 day


        user_edges = model_info.get("bin_edges", None)
        user_labels = model_info.get("bin_labels", None)
        user_pos = model_info.get("positive_bin", None)

        if user_edges is None:
            # Compute threshold from TRAIN ONLY to avoid leakage
            q = float(model_info.get("bin_quantile", 0.90))  # allow override, default 0.90
            thr = np.quantile(y_train, q)
            thr = int(np.rint(thr))  # round to nearest day

            # Guard: ensure both classes exist; if degenerate, nudge the quantile
            if (y_train >= thr).sum() == 0 or (y_train < thr).sum() == 0:
                for q_try in (0.85, 0.95, 0.80, 0.97):
                    thr_try = int(np.rint(np.quantile(y_train, q_try)))
                    if (y_train >= thr_try).sum() > 0 and (y_train < thr_try).sum() > 0:
                        thr = thr_try
                        q = q_try
                        break

            # With right=False, bins are [a, b) so value == thr goes to the second bin
            bin_edges  = [-np.inf, thr, np.inf]
            bin_labels = [0, 1]       # 0 = shorter, 1 = extended
            positive_bin = 1
        else:
            bin_edges  = user_edges
            bin_labels = user_labels if user_labels is not None else [0, 1]
            positive_bin = user_pos if user_pos is not None else 1
            thr = bin_edges[1] if len(bin_edges) == 3 else None  # for logging only

        # Apply binning (>= thr is positive because right=False => [thr, inf))
        ytr_bin = pd.cut(y_train, bins=bin_edges, labels=bin_labels, right=False).astype(int)
        yte_bin = pd.cut(y_test,  bins=bin_edges, labels=bin_labels, right=False).astype(int)

        print(f">>> Binning: edges={bin_edges} (thr={thr}), labels={bin_labels}; "
            f"class counts train: {ytr_bin.value_counts().to_dict()}, "
            f"test: {yte_bin.value_counts().to_dict()}")

        # Sanity
        y_type = type_of_target(ytr_bin)
        if y_type not in ("binary",):
            raise ValueError(f"Expected binary after binning, got {y_type}. "
                            f"Check edges={bin_edges}, labels={bin_labels}")

        # ----- 2) Pipeline (classifier required) -----
        clf = model_info["classifier"]
        if not is_classifier(clf):
            raise TypeError(f"{model_name} must be a classifier, got {type(clf).__name__}")

        pipe = Pipeline([("classifier", clf)])

        # Normalize param grid keys to 'classifier__'
        raw_grid = model_info.get("param_grid", {})
        param_grid = {}
        for k, v in raw_grid.items():
            if k.startswith("classifier__") or "__" in k:
                param_grid[k] = v
            else:
                param_grid["classifier__" + k] = v

        # Add class_weight='balanced' if supported and not provided
        if "class_weight" in clf.get_params() and "classifier__class_weight" not in param_grid:
            param_grid["classifier__class_weight"] = ["balanced"]

        # ----- 3) Scoring & CV -----
        if not isinstance(cv_strategy, StratifiedKFold):
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scoring = {
            "f1":        make_scorer(f1_score, average="binary", zero_division=0),
            "accuracy":  "accuracy",
            "precision": make_scorer(precision_score, average="binary", zero_division=0),
            "recall":    make_scorer(recall_score, average="binary", zero_division=0),
            "roc_auc":   "roc_auc",
        }

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit="f1",  # optimize F1 in CV; we also report ROC AUC on holdout
            cv=cv_strategy,
            n_jobs=-1,
            verbose=1,
        )

        print(f">>> Fitting grid search for {model_name} (binary classification)...")
        grid_search.fit(X_train, ytr_bin)
        best_model = grid_search.best_estimator_
        print(">>> Completed grid search.")
        print(f">>> Best parameters: {grid_search.best_params_}")

        # ----- 4) Holdout evaluation -----
        y_pred = best_model.predict(X_test)

        # Probability for positive class (needed for ROC AUC & threshold tuning)
        proba = None
        pos_auc = None
        cls = best_model.named_steps["classifier"]
        if hasattr(cls, "predict_proba"):
            probs = cls.predict_proba(X_test)
            # classes_ is [0,1] but be safe:
            classes_ = cls.classes_
            pos_idx = int(np.where(classes_ == positive_bin)[0][0])
            proba = probs[:, pos_idx]
            pos_auc = roc_auc_score(yte_bin, proba)

        acc  = accuracy_score(yte_bin, y_pred)
        f1b  = f1_score(yte_bin, y_pred, average="binary", zero_division=0)
        prec = precision_score(yte_bin, y_pred, average="binary", zero_division=0)
        rec  = recall_score(yte_bin, y_pred, average="binary", zero_division=0)
        cm   = confusion_matrix(yte_bin, y_pred, labels=bin_labels)
        rep  = classification_report(yte_bin, y_pred, zero_division=0, digits=3)

        # ----- 5) Threshold tuning (optional but useful) -----
        tuned = None
        if proba is not None:
            best = {"thr": 0.5, "f1": f1b, "acc": acc, "rec": rec, "prec": prec, "auc": pos_auc, "cm": cm}
            for t in np.linspace(0.2, 0.8, 25):
                pred_t = (proba >= t).astype(int)
                f1_t   = f1_score(yte_bin, pred_t, zero_division=0)
                if f1_t > best["f1"]:
                    best["thr"]  = t
                    best["f1"]   = f1_t
                    best["acc"]  = accuracy_score(yte_bin, pred_t)
                    best["rec"]  = recall_score(yte_bin, pred_t, zero_division=0)
                    best["prec"] = precision_score(yte_bin, pred_t, zero_division=0)
                    best["auc"]  = roc_auc_score(yte_bin, proba)  # AUC unaffected by threshold
                    best["cm"]   = confusion_matrix(yte_bin, pred_t, labels=bin_labels)
            tuned = best
            print(f">>> Tuned(>1d) | thr={best['thr']:.2f}  F1={best['f1']:.3f}  "
                f"Acc={best['acc']:.3f}  Rec={best['rec']:.3f}  Prec={best['prec']:.3f}")

        # ----- 6) Metrics (backward-compatible keys) -----
        metrics = {
            "bins": {"edges": bin_edges, "labels": bin_labels, "positive_bin": positive_bin},
            "best_params": grid_search.best_params_,
            "holdout": {
                "accuracy": acc,
                "f1": f1b,
                "precision": prec,
                "recall": rec,
                "roc_auc": pos_auc,                 # may be None if model lacks predict_proba
                "confusion_matrix": cm,
                "report": rep,
            },
            "threshold_tuned": tuned,
        }
        # Keep your caller happy:
        metrics["auc"] = pos_auc if pos_auc is not None else f1b  # selection metric
        metrics["confusion_matrix"] = cm

        print(f">>> Holdout | Acc={acc:.3f}  F1={f1b:.3f}  "
            f"Prec={prec:.3f}  Rec={rec:.3f}  "
            f"AUC={(pos_auc if pos_auc is not None else float('nan')):.3f}")
        return best_model, metrics

    def compute_shap_values(self, best_estimator, X_test):
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
        
        print(">>> SHAP computation complete.")
        return {"shap_values": shap_values, "shap_summary": shap_summary_df}

    def save_results_to_csv(self, filename, best_model_name, best_metrics, shap_data, target, results_folder):
        """
        Saves performance metrics and SHAP summary to CSV files.
        Filenames include the target and a timestamp for easy tracking.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        performance_csv = os.path.join(results_folder, f"performance_{filename}_{best_model_name}_{target}_{timestamp}.csv")
        shap_csv = os.path.join(results_folder, f"shap_summary_{best_model_name}_{target}_{timestamp}.csv")
        
        # Save performance metrics.
        perf_df = pd.DataFrame([best_metrics])
        perf_df.to_csv(performance_csv, index=False)
        print(f"Saved performance metrics to {performance_csv}")
        
        # Save SHAP summary.
        shap_data["shap_summary"].to_csv(shap_csv, index=False)
        print(f"Saved SHAP summary to {shap_csv}")

    def show_shap_beeswarm(self, shap_data, X_test, show_plot=True, save_plot=False, plot_filename=None):
        """
        Displays or saves a SHAP beeswarm plot based on the provided shap_data.
        
        Parameters:
            shap_data (dict): Dictionary returned by compute_shap_values() 
                            containing 'shap_values' and 'shap_summary'.
            X_test (pd.DataFrame): The test set used to compute shap_values.
            show_plot (bool): Whether to display the beeswarm plot.
            save_plot (bool): Whether to save the beeswarm plot to a file.
            plot_filename (str): Name of the file to save the plot. If None, 
                                defaults to 'shap_beeswarm_plot.png'.
        """
        import shap
        import matplotlib.pyplot as plt

        shap_values = shap_data["shap_values"]

        # Create a new figure (or you can control figure size here if you want).
        plt.figure()

        # Pass show=False so that shap.summary_plot doesn't immediately display the plot.
        shap.summary_plot(shap_values, X_test, max_display=100, plot_type="dot", show=False)

        if save_plot:
            if not plot_filename:
                plot_filename = "shap_beeswarm_plot.png"
            plt.savefig(plot_filename, bbox_inches='tight')
            print(f">>> SHAP beeswarm plot saved as '{plot_filename}'.")

        if show_plot:
            plt.show()

        # Close the figure so it doesn't hang or cause issues in certain environments.
        plt.close()
    def show_shap_dependence_plot(
        self,
        shap_data,
        X,
        feature: str,
        interaction_feature: str | None = None,
        show_plot: bool = True,
        save_plot: bool = False,
        plot_filename: str | None = None,
        dpi: int = 200,
        # NEW: optional axis limits & feature clipping for plotting only
        xlim: tuple[float | None, float | None] | None = None,
        clip_limits: dict[str, tuple[float | None, float | None]] | None = None,
    ):
        """
        SHAP dependence plot with optional X clipping and x-axis limits.
        """
        import shap
        import numpy as np
        import matplotlib.pyplot as plt

        shap_values = shap_data["shap_values"]

        # --- make a plotting copy and clip if requested ---
        X_plot = X.copy()
        if clip_limits:
            for col, (lo, hi) in clip_limits.items():
                if col in X_plot.columns:
                    if lo is not None:
                        X_plot[col] = np.maximum(X_plot[col], lo)
                    if hi is not None:
                        X_plot[col] = np.minimum(X_plot[col], hi)

        plt.figure()
        shap.dependence_plot(
            feature,
            shap_values,
            X_plot,
            interaction_index=interaction_feature,
            show=False,
        )

        # Force x-axis limits if provided
        if xlim is not None:
            plt.xlim(xlim)

        if save_plot:
            if not plot_filename:
                plot_filename = f"shap_dependence_{feature}_by_{interaction_feature or 'none'}.png"
            plt.savefig(plot_filename, bbox_inches="tight", dpi=dpi)
            print(f">>> Saved SHAP dependence: {plot_filename}")

        if show_plot:
            plt.show()

        plt.close()
    
    def save_explanatory_plots(
        self,
        shap_data,
        X_test,
        pipeline,                   # trained best_pipeline (with the classifier)
        out_dir: str,
        prefix: str,
        pairs: list[tuple[str,str]] = (("anetime","optime"), ("optime","anetime")),
        save_shap: bool = True,
        save_pdp_ice: bool = True,
        save_pdp_2d: bool = True,
        dpi: int = 200,
        # NEW: cap 'anetime' at 600 min (you can extend this dict as needed)
        cap_optime_at = 600.0,
    ):
        """
        Saves a bundle of model explanation plots with optional capping for 'optime'.
        Files: {out_dir}/{prefix}_<plotname>.png
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.inspection import PartialDependenceDisplay

        os.makedirs(out_dir, exist_ok=True)

        # --- plotting copy with optional cap on anetime ---
        X_plot = X_test.copy()
        if cap_optime_at is not None and "optime" in X_plot.columns:
            X_plot["optime"] = np.minimum(X_plot["optime"], cap_optime_at)

        # --- SHAP dependence plots ---
        if save_shap:
            # pass clip + xlim so the scatter never runs past 600 on the x-axis
            clip_dict = {"optime": (None, cap_optime_at)} if cap_optime_at is not None else None
            for f, inter in pairs:
                fn = os.path.join(out_dir, f"{prefix}_shapdep_{f}_by_{inter}.png")
                self.show_shap_dependence_plot(
                    shap_data=shap_data,
                    X=X_plot,
                    feature=f,
                    interaction_feature=inter,
                    show_plot=False,
                    save_plot=True,
                    plot_filename=fn,
                    dpi=dpi,
                    xlim=(None, cap_optime_at) if f == "optime" and cap_optime_at is not None else None,
                    clip_limits=clip_dict,   # only affects plotting copy
                )

        # Collect unique features referenced
        feat_set = sorted(set([p[0] for p in pairs]) | set([p[1] for p in pairs]))



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

        columns_to_drop_if_empty = ["ped_sap_name1"]
        data = ScoliosisFeatureEngineeringService.remove_empty_rows(data, columns_to_drop_if_empty)
        data = ScoliosisFeatureEngineeringService.replace_with_nan(data, missing_tokens=["-99", "null", "#null!", "na", "n/a", "", " "])
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
        data[numeric_cols] = data[numeric_cols].clip(np.finfo(np.float32).min,np.finfo(np.float32).max)

        data = ScoliosisFeatureEngineeringService.make_rf_compatible(data)
        data = ScoliosisFeatureEngineeringService.rename_for_xgb_compatibility(data)
        
        # FIX: Ensure target column is numeric
        if target_col in data.columns:
            data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
            # Remove rows where target is NaN
            before_count = len(data)
            data = data.dropna(subset=[target_col])
            after_count = len(data)
            if before_count > after_count:
                print(f"Dropped {before_count - after_count} rows with invalid target values")
            data[target_col] = data[target_col].astype(int)
    
        for col in data.select_dtypes(include=['object']).columns:
            print(f"Warning: Column {col} is still object type")

            # Verify target is numeric
            if target_col in data.columns:
                data[target_col] = pd.to_numeric(data[target_col], errors='coerce').fillna(0).astype(int)

        return data


class ScoliosisFeatureEngineeringService:
    @staticmethod
    def replace_with_nan(data, missing_tokens=["-99", "null", "#null!", "na", "n/a", "", " "]):
        if missing_tokens is None:
            missing_tokens = ["-99", "null", "#null!", "na", "n/a", "", " "]
        
        for col in data.columns:
            # Check if column is already numeric
            if pd.api.types.is_numeric_dtype(data[col]):
                # For numeric columns, only replace -99 and similar numeric markers
                data[col] = data[col].replace([-99, -99.0, -1], np.nan)
            else:
                # For non-numeric columns, convert to string and process
                data[col] = data[col].astype(str).str.strip()
                for token in missing_tokens:
                    pattern = rf"(?i)^{re.escape(token)}$"
                    data.loc[data[col].str.match(pattern, na=False), col] = np.nan
        
        return data


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
        bmi_flags = [0, 1, 2]  # 0: invalid_extreme, 1: invalid_clinical, 2: unusual_clinical
        df['bmi_quality_flag'] = np.select(
            condlist=bmi_conditions,
            choicelist=bmi_flags,
            default=3  # 3: valid
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
        data = data.drop(columns="los_category", errors="ignore")
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
        # First, convert any remaining object columns
        object_cols = data.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"Converting object columns to numeric: {list(object_cols)}")
            for col in object_cols:
                # Special handling for diagnostic codes - might want to drop or encode differently
                if col in ['podiag10', 'podiagtx10']:
                    # Option 1: Drop them
                    data = data.drop(columns=[col])
                    continue
                    # Option 2: Create a simpler encoding (e.g., just presence/absence)
                    # data[col + '_present'] = data[col].notna().astype(int)
                    # data = data.drop(columns=[col])
                else:
                    # Try to convert to numeric first
                    try:
                        data[col] = pd.to_numeric(data[col], errors='raise')
                    except (ValueError, TypeError):
                        # If it's truly categorical, encode it
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        # Handle NaN values
                        data[col] = data[col].fillna('missing')
                        data[col] = le.fit_transform(data[col].astype(str))
        
        # Then handle missing values in numeric columns
        missing_cols = data.columns[data.isna().any()].tolist()
        print(f"Columns with missing values: {missing_cols}")
        
        for col in missing_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())
            elif pd.api.types.is_bool_dtype(data[col]):
                data[col] = data[col].fillna(False).astype(int)
            else:
                data[col] = data[col].fillna(0)
        
        # Final verification
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
