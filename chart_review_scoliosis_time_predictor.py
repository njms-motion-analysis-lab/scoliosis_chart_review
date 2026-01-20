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
    classification_report, make_scorer, average_precision_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.utils.multiclass import type_of_target

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
        X = data.drop(columns=[target_column])
        y = data[target_column]

        if bin_string is not None:
            y = self.convert_to_binary(data=X, target=y, statement=bin_string)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,          # critical for rare outcomes
            shuffle=True
        )
        return X_train, X_test, y_train, y_test

    def grid_search_pipeline(
        self,
        data,
        target_column="tothlos",  # Change as needed.
        test_size=0.2,
        random_state=42,
        models=None,
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        bin_string=None,
        return_all_models=False,
        use_smote=True,
        sampling_strategy="borderline",
        filter_features=True,
        top_n_features=None,
        min_feature_importance=0.001,
        n_jobs=-1  # Control parallelism; set to 2-4 to limit memory usage
    ):
        """
        Splits data, optionally applies feature filtering, performs grid searches
        for all models.

        If return_all_models=True, returns a dict of all model results.
        Otherwise, returns only the best model (backward compatible).

        Args:
            use_smote: If True, apply oversampling to handle class imbalance (default: True)
            sampling_strategy: Which sampler to use - "smote", "borderline", "adasyn" (default: "borderline")
            filter_features: If True, filter low-importance features (default: True)
            top_n_features: If set, keep only top N features (overrides min_feature_importance)
            min_feature_importance: Remove features below this importance threshold (default: 0.001)
        """
        print("=== Starting grid search pipeline ===")
        if models is None or not isinstance(models, dict):
            raise ValueError("You must provide a valid dictionary of models and param grids.")

        print("Preparing training and testing sets...")
        X_train, X_test, y_train, y_test = self.prepare_train_test(
            data, target_column, test_size, random_state, bin_string=bin_string
        )

        # Feature filtering (removes obviously useless features)
        # TODO: Stephen, skip feature filter for now
        # if filter_features:
        #     X_train, X_test = self._filter_low_importance_features(
        #         X_train, y_train, X_test,
        #         min_importance_threshold=min_feature_importance,
        #         top_n=top_n_features,
        #         n_jobs=n_jobs
        #     )
        # else:
        #     print("Skipping feature filtering (filter_features is False).")

        best_estimator = None
        best_score = -np.inf
        best_model_name = None
        best_metrics = {}

        # Store all model results
        all_results = {}

        for model_name, model_info in models.items():
            print(f"\n--- Starting grid search for {model_name} ---")
            try:
                estimator, metrics = self._run_grid_search(model_name, model_info, X_train, y_train, X_test, y_test, cv_strategy, use_smote=use_smote, sampling_strategy=sampling_strategy, n_jobs=n_jobs)
                all_results[model_name] = {
                    "estimator": estimator,
                    "metrics": metrics
                }
                # Use PR-AUC for model selection (better for imbalanced data)
                selection_metric = metrics.get("cv_pr_auc", -np.inf)
                if selection_metric > best_score:
                    best_score = selection_metric
                    best_estimator = estimator
                    best_model_name = model_name
                    best_metrics = metrics
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                print("Continuing to next model...")
                continue

        print(f"\n=== Best Overall Model: {best_model_name} ===")
        print(f"Final PR-AUC: {best_metrics.get('pr_auc', 0):.3f} (ROC-AUC: {best_metrics.get('auc', 0):.3f})")
        print("Final Confusion Matrix:")
        print(best_metrics["confusion_matrix"])

        if return_all_models:
            return all_results, best_model_name, X_test, y_test

        # Backward compatible return
        return best_estimator, best_metrics, best_model_name, X_test

    def _filter_low_importance_features(self, X_train, y_train, X_test, min_importance_threshold=0.001, top_n=None, n_jobs=-1):
        """
        Quick feature filtering using a fast Random Forest to estimate feature importance.
        Removes features with very low importance before main model training.

        Args:
            min_importance_threshold: Remove features with importance below this (default: 0.001)
            top_n: If set, keep only the top N most important features (overrides threshold)
            n_jobs: Number of parallel jobs (-1 for all cores, 2-4 recommended to limit memory)
        """
        print(">>> Running quick feature importance filtering...")

        # Train a quick RF to get feature importances
        quick_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=n_jobs,
            class_weight='balanced'
        )
        quick_rf.fit(X_train, y_train)

        # Get feature importances
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': quick_rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f">>> Top 10 features by importance:")
        print(importances.head(10).to_string(index=False))

        # Filter features
        if top_n is not None:
            selected_features = importances.head(top_n)['feature'].tolist()
            print(f">>> Keeping top {top_n} features")
        else:
            selected_features = importances[importances['importance'] >= min_importance_threshold]['feature'].tolist()
            print(f">>> Keeping {len(selected_features)} features with importance >= {min_importance_threshold}")

        removed_count = len(X_train.columns) - len(selected_features)
        print(f">>> Removed {removed_count} low-importance features")

        if selected_features:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]

        return X_train, X_test

    def _run_grid_search(self, model_name, model_info, X_train, y_train, X_test, y_test, cv_strategy, use_smote=True, sampling_strategy="borderline", n_jobs=-1):
        """
        Binary classification. Handles both:
        - Already-binary targets (e.g., reoperation with values 0/1)
        - Continuous targets that need binning (e.g., LOS)
        Returns metrics with 'auc' key (ROC AUC).

        Args:
            use_smote: If True, apply oversampling within CV folds to handle class imbalance
            sampling_strategy: Which sampler to use - "smote", "borderline", "adasyn" (default: "borderline")
        """

        # ----- 1) Check if target is already binary -----
        unique_train = set(y_train.unique())
        unique_test = set(y_test.unique())
        is_already_binary = unique_train.issubset({0, 1, 0.0, 1.0}) and unique_test.issubset({0, 1, 0.0, 1.0})

        if is_already_binary:
            # Target is already binary - use as-is
            print(f">>> Target is already binary (values: {unique_train}), skipping binning")
            ytr_bin = y_train.astype(int)
            yte_bin = y_test.astype(int)
            bin_edges = None
            bin_labels = [0, 1]
            positive_bin = 1
            thr = None
            print(f">>> Class counts train: {ytr_bin.value_counts().to_dict()}, "
                  f"test: {yte_bin.value_counts().to_dict()}")
        else:
            # Continuous target - apply binning
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

        # Check if classifier handles imbalance internally (e.g., BalancedRandomForestClassifier)
        clf_type = type(clf).__name__
        handles_imbalance_internally = clf_type in ['BalancedRandomForestClassifier', 'BalancedBaggingClassifier']

        # Use imbalanced-learn Pipeline with oversampling for class imbalance handling
        # Skip SMOTE for classifiers that handle imbalance internally
        if use_smote and not handles_imbalance_internally:
            if sampling_strategy == "borderline":
                print(f">>> Using BorderlineSMOTE oversampling for {model_name}")
                sampler = BorderlineSMOTE(random_state=42, k_neighbors=5)
            elif sampling_strategy == "adasyn":
                print(f">>> Using ADASYN oversampling for {model_name}")
                sampler = ADASYN(random_state=42, n_neighbors=5)
            else:  # default to regular SMOTE
                print(f">>> Using SMOTE oversampling for {model_name}")
                sampler = SMOTE(random_state=42, k_neighbors=3)

            pipe = ImbPipeline([
                ("sampler", sampler),
                ("classifier", clf)
            ])
        else:
            if handles_imbalance_internally:
                print(f">>> {model_name} handles class imbalance internally - skipping SMOTE")
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
        # BUT skip for classifiers that handle imbalance internally (they balance via sampling)
        if "class_weight" in clf.get_params() and "classifier__class_weight" not in param_grid:
            if not handles_imbalance_internally:
                param_grid["classifier__class_weight"] = ["balanced"]
            else:
                print(f">>> Skipping class_weight for {model_name} (already handles imbalance via sampling)")

        # ----- 3) Scoring & CV -----
        # Accept both StratifiedKFold and RepeatedStratifiedKFold
        if not isinstance(cv_strategy, (StratifiedKFold, RepeatedStratifiedKFold)):
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scoring = {
            "f1": "f1",
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",  # safest and standard
        }
        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit="pr_auc",  # Optimize PR-AUC - more meaningful for imbalanced data
            cv=cv_strategy,
            n_jobs=n_jobs,
            verbose=1,
        )

        print(f">>> Fitting grid search for {model_name} (binary classification)...")
        grid_search.fit(X_train, ytr_bin)
        best_model = grid_search.best_estimator_
        cv_pr_auc = grid_search.best_score_
        print(">>> Completed grid search.")
        print(f">>> Best parameters: {grid_search.best_params_}")

        # ----- 4) Holdout evaluation -----
        y_pred = best_model.predict(X_test)

        # Probability for positive class (needed for ROC AUC, PR-AUC & threshold tuning)
        proba = None
        pos_auc = None
        pr_auc = None
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(X_test)
            proba = probs[:, 1]  # for binary 0/1
            pos_auc = roc_auc_score(yte_bin, proba)
            pr_auc = average_precision_score(yte_bin, proba)  # More informative for imbalanced data

        acc  = accuracy_score(yte_bin, y_pred)
        f1b  = f1_score(yte_bin, y_pred, average="binary", zero_division=0)
        prec = precision_score(yte_bin, y_pred, average="binary", zero_division=0)
        rec  = recall_score(yte_bin, y_pred, average="binary", zero_division=0)
        cm   = confusion_matrix(yte_bin, y_pred, labels=bin_labels)
        rep  = classification_report(yte_bin, y_pred, zero_division=0, digits=3)

        # ----- 5) Threshold tuning (optional but useful) -----
        oof_proba = cross_val_predict(
            best_model,
            X_train, ytr_bin,
            cv=cv_strategy,
            method="predict_proba",
            n_jobs=n_jobs
        )[:, 1]

        # choose threshold based on training OOF predictions
        best_thr, best_f1 = 0.5, -1
        thr_grid = np.linspace(0.01, 0.50, 100)
        for t in thr_grid:
            pred_t = (oof_proba >= t).astype(int)
            f1_t = f1_score(ytr_bin, pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1, best_thr = f1_t, t

        # now evaluate on test using that fixed threshold
        test_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred_thr = (test_proba >= best_thr).astype(int)
        

        
        tuned = None
        if proba is not None:
            best = {"thr": 0.5, "f1": f1b, "acc": acc, "rec": rec, "prec": prec, "auc": pos_auc, "cm": cm}
            tuned = {
                "thr": float(best_thr),
                "f1": float(f1_score(yte_bin, y_pred_thr, zero_division=0)),
                "acc": float(accuracy_score(yte_bin, y_pred_thr)),
                "rec": float(recall_score(yte_bin, y_pred_thr, zero_division=0)),
                "prec": float(precision_score(yte_bin, y_pred_thr, zero_division=0)),
                "train_oof_f1": float(best_f1),
                "confusion_matrix": confusion_matrix(yte_bin, y_pred_thr, labels=bin_labels),
            }

            print(
                f">>> Tuned(threshold from OOF) | thr={tuned['thr']:.2f} "
                f"OOF-F1={tuned['train_oof_f1']:.3f} "
                f"Test-F1={tuned['f1']:.3f} "
                f"Acc={tuned['acc']:.3f} "
                f"Rec={tuned['rec']:.3f} "
                f"Prec={tuned['prec']:.3f}"
            )

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
                "pr_auc": pr_auc,                   # Precision-Recall AUC (better for imbalanced data)
                "confusion_matrix": cm,
                "report": rep,
            },
            "threshold_tuned": tuned,
            # For ROC/PR curve generation
            "y_proba": proba,
            "y_test": yte_bin,
        }
        # Keep your caller happy:
        metrics["auc"] = pos_auc if pos_auc is not None else f1b  # selection metric
        metrics["pr_auc"] = pr_auc  # Also expose at top level
        metrics["confusion_matrix"] = cm
        metrics["cv_pr_auc"] = cv_pr_auc

        print(f">>> Holdout | Acc={acc:.3f}  F1={f1b:.3f}  "
            f"Prec={prec:.3f}  Rec={rec:.3f}  "
            f"ROC-AUC={(pos_auc if pos_auc is not None else float('nan')):.3f}  "
            f"PR-AUC={(pr_auc if pr_auc is not None else float('nan')):.3f}")
        return best_model, metrics

    def compute_shap_values(self, best_estimator, X_test):
        """
        Computes SHAP values for the best estimator and returns a dictionary containing:
        - shap_values: array of shape (n_samples, n_features) for the positive class when applicable
        - shap_summary: DataFrame of mean absolute SHAP values per feature.

        Handles common cases:
        - LogisticRegression (LinearExplainer)
        - Tree-based models (TreeExplainer)
        - XGBClassifier with base_score bracket workaround
        """
        print(">>> Computing SHAP values...")

        # Your pipelines use the step name 'classifier'
        best_classifier = best_estimator.named_steps["classifier"]

        model_type = type(best_classifier).__name__
        print(f">>> Model type: {model_type}")

        # --- choose explainer / compute raw shap values ---
        if model_type == "LogisticRegression":
            print(">>> Using LinearExplainer for linear model")
            explainer = shap.LinearExplainer(best_classifier, X_test)
            raw_shap_values = explainer.shap_values(X_test)

        elif model_type == "BalancedRandomForestClassifier":
            print(">>> Using TreeExplainer for BalancedRandomForestClassifier")
            explainer = shap.TreeExplainer(best_classifier)
            raw_shap_values = explainer.shap_values(X_test)

        elif model_type == "XGBClassifier":
            print(">>> Using TreeExplainer for XGBoost (with base_score fix)")
            import builtins
            import importlib
            import shap.explainers._tree as tree_module

            X_test_np = X_test.values.astype(np.float64)

            original_float = builtins.float

            def safe_float(val):
                if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                    val = val[1:-1]
                return original_float(val)

            builtins.float = safe_float
            try:
                importlib.reload(tree_module)
                explainer = shap.TreeExplainer(best_classifier)
                raw_shap_values = explainer.shap_values(X_test_np)
            finally:
                builtins.float = original_float

        else:
            print(">>> Using TreeExplainer for tree-based model")
            explainer = shap.TreeExplainer(best_classifier)
            raw_shap_values = explainer.shap_values(X_test)

        # --- normalize output to (n_samples, n_features) ---
        # Common binary-classification outputs:
        #  - list of two arrays: [class0, class1] -> use class1
        #  - array (n, p, 2) -> use [:, :, 1]
        #  - array (n, p) -> use as-is
        if isinstance(raw_shap_values, list):
            if len(raw_shap_values) >= 2:
                shap_values = raw_shap_values[1]  # positive class
            else:
                shap_values = raw_shap_values[0]
        else:
            raw_shape = np.array(raw_shap_values).shape
            if len(raw_shape) == 3 and raw_shape[2] >= 2:
                shap_values = raw_shap_values[:, :, 1]  # positive class
            else:
                shap_values = raw_shap_values

        # Ensure numpy array for downstream ops
        shap_values = np.asarray(shap_values)

        # --- summary ---
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_summary_df = pd.DataFrame(
            {"feature": X_test.columns, "mean_abs_shap": mean_abs_shap}
        ).sort_values(by="mean_abs_shap", ascending=False)

        print(">>> SHAP computation complete.")
        return {"shap_values": shap_values, "shap_summary": shap_summary_df}

    def save_results_to_csv(self, best_metrics, shap_data, target, result_paths):
        """
        Saves performance metrics and SHAP summary to CSV files.
        Files are saved to organized subfolders with timestamps.

        Args:
            best_metrics: Dictionary of performance metrics
            shap_data: Dictionary containing SHAP values and summary
            target: Target column name
            result_paths: Dictionary with paths for 'performance_summaries' and 'shap_summaries'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        performance_csv = os.path.join(result_paths["performance_summaries"], f"{target}_{timestamp}.csv")
        shap_csv = os.path.join(result_paths["shap_summaries"], f"{target}_{timestamp}.csv")

        # Flatten metrics for easy Excel viewing
        holdout = best_metrics.get("holdout", {})
        tuned = best_metrics.get("threshold_tuned", {})
        cm = best_metrics.get("confusion_matrix")

        # Extract confusion matrix values (assuming binary: TN, FP, FN, TP)
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm is not None:
            try:
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]
            except (IndexError, TypeError):
                pass

        # Build flattened row with key metrics first for easy viewing
        flat_metrics = {
            # Key metrics at the front (holdout set performance)
            "accuracy": holdout.get("accuracy"),
            "f1": holdout.get("f1"),
            "roc_auc": holdout.get("roc_auc"),
            "pr_auc": holdout.get("pr_auc"),  # PR-AUC is more informative for imbalanced data
            "precision": holdout.get("precision"),
            "recall": holdout.get("recall"),
            # Confusion matrix broken out
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            # Threshold-tuned metrics (if available)
            "tuned_threshold": tuned.get("thr") if tuned else None,
            "tuned_f1": tuned.get("f1") if tuned else None,
            "tuned_accuracy": tuned.get("acc") if tuned else None,
            "tuned_precision": tuned.get("prec") if tuned else None,
            "tuned_recall": tuned.get("rec") if tuned else None,
            # Model parameters (as string for reference)
            "best_params": str(best_metrics.get("best_params", {})),
        }

        perf_df = pd.DataFrame([flat_metrics])
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
        # Convert ordinal columns (like ASA Class) to numeric before categorical encoding
        data = ScoliosisFeatureEngineeringService.encode_ordinal_columns(data)
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
        data["bmi"] = np.where(df["bmi_quality_flag"].isin([2, 3]), df["bmi"], np.nan)
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
        ASA Class is included for one-hot encoding.
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
    def encode_ordinal_columns(data):
        """
        Encodes ordinal columns like ASA class into simple numeric dummy variables.
        Extracts the ASA class number from messy strings and creates clean columns:
        - ASA_1: patient is ASA class 1
        - ASA_2: patient is ASA class 2
        - ASA_3plus: patient is ASA class 3, 4, or 5
        """
        # Handle ASA class column
        asa_col = None
        for col in data.columns:
            if 'asaclas' in col.lower():
                asa_col = col
                break

        if asa_col is not None:
            print(f"Processing ASA class column: {asa_col}")

            # Extract the ASA class number (1, 2, 3, 4, 5) from the string values
            def extract_asa_class(value):
                if pd.isna(value):
                    return None
                val_str = str(value).strip()
                # Look for "ASA 1", "ASA 2", etc. pattern
                match = re.search(r'ASA\s*(\d)', val_str, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                # Also try to find just a single digit if it's already numeric
                try:
                    num = int(float(val_str))
                    if 1 <= num <= 5:
                        return num
                except (ValueError, TypeError):
                    pass
                return None

            # Extract ASA class numbers
            asa_values = data[asa_col].apply(extract_asa_class)

            # Get unique non-null ASA classes found in the data
            unique_classes = sorted([c for c in asa_values.dropna().unique() if c is not None])
            print(f"ASA classes found: {unique_classes}")

            # Create binary columns: ASA_1, ASA_2, ASA_3plus (3, 4, or 5)
            data["ASA_1"] = (asa_values == 1).astype(int)
            data["ASA_2"] = (asa_values == 2).astype(int)
            data["ASA_3plus"] = (asa_values >= 3).astype(int)

            print(f"  Created ASA_1: {data['ASA_1'].sum()} cases")
            print(f"  Created ASA_2: {data['ASA_2'].sum()} cases")
            print(f"  Created ASA_3plus: {data['ASA_3plus'].sum()} cases")

            # Drop the original ASA column to prevent it from being one-hot encoded again
            data = data.drop(columns=[asa_col])
            print(f"Dropped original column: {asa_col}")

        return data

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
