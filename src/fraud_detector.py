"""
Multi-feature fraud detection.

Two detection strategies:
  1. **Unsupervised** – Isolation Forest (when no ground-truth labels exist).
  2. **Supervised** – Random Forest with SMOTE over-sampling and
     class-weight balancing (when IS_FRAUD labels are available).

Common functionality:
  - Engineers **9 claim-level features** from row-level data.
  - Scales features with StandardScaler.
  - Proper train / test split.
  - Reports precision, recall, F1, accuracy, plus unsupervised metrics.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler

import config as cfg

logger = logging.getLogger("insurance_ml.fraud")


class FraudDetector:
    """Anomaly-based fraud detection on insurance claims."""

    def __init__(self) -> None:
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    # ── Feature Engineering ──────────────────────────────────────────────

    @staticmethod
    def engineer_features(merged_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate row-level data into claim-level features."""
        # Resolve amount column
        amount_col = cfg.AMOUNT_COL
        if amount_col not in merged_df.columns:
            amount_cols = [c for c in merged_df.columns if "AMOUNT" in c.upper()]
            if not amount_cols:
                raise ValueError("No amount column found")
            amount_col = amount_cols[0]

        features = (
            merged_df.groupby(cfg.CLAIM_ID_COL)
            .agg(
                total_amount=(amount_col, "sum"),
                part_count=(amount_col, "count"),
                avg_part_amount=(amount_col, "mean"),
                max_part_amount=(amount_col, "max"),
                min_part_amount=(amount_col, "min"),
                std_part_amount=(amount_col, "std"),
            )
            .reset_index()
        )

        # Fill NaN std for single-part claims
        features["std_part_amount"] = features["std_part_amount"].fillna(0)

        # Derived features
        features["amount_range"] = (
            features["max_part_amount"] - features["min_part_amount"]
        )
        features["coeff_variation"] = np.where(
            features["avg_part_amount"] > 0,
            features["std_part_amount"] / features["avg_part_amount"],
            0,
        )
        features["log_total_amount"] = np.log1p(features["total_amount"])

        logger.info(
            f"Engineered {len(features.columns) - 1} features "
            f"for {len(features)} claims"
        )
        return features

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, claim_features: pd.DataFrame) -> "FraudDetector":
        """Train the Isolation Forest on the feature matrix."""
        self.feature_names = [
            c for c in claim_features.columns if c != cfg.CLAIM_ID_COL
        ]
        X = claim_features[self.feature_names].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=cfg.ISO_FOREST_N_ESTIMATORS,
            contamination=cfg.ISO_FOREST_CONTAMINATION,
            max_features=cfg.ISO_FOREST_MAX_FEATURES,
            bootstrap=cfg.ISO_FOREST_BOOTSTRAP,
            random_state=cfg.RANDOM_STATE,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        logger.info(
            f"Fraud detector trained on {X.shape[0]} claims × {X.shape[1]} features"
        )
        return self

    # ── Prediction ───────────────────────────────────────────────────────

    def predict(self, claim_features: pd.DataFrame) -> pd.DataFrame:
        """Return predictions with anomaly labels and scores."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted – call fit() first")

        feat_cols = [c for c in self.feature_names if c in claim_features.columns]
        X = claim_features[feat_cols].values
        X_scaled = self.scaler.transform(X)

        results = claim_features.copy()
        results["anomaly_label"] = self.model.predict(X_scaled)  # 1=normal, -1=anomaly
        results["anomaly_score"] = self.model.decision_function(X_scaled)
        results["is_fraud"] = (results["anomaly_label"] == -1).astype(int)

        n_fraud = int(results["is_fraud"].sum())
        logger.info(
            f"Predicted: {n_fraud} suspicious claims "
            f"({n_fraud / len(results) * 100:.1f}%)"
        )
        return results

    # ── Evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        claim_features: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> Dict:
        """Evaluate with unsupervised cluster-quality metrics."""
        feat_cols = [c for c in self.feature_names if c in claim_features.columns]
        X_scaled = self.scaler.transform(claim_features[feat_cols].values)

        labels = predictions["anomaly_label"].values

        metrics: Dict = {
            "total_claims": len(predictions),
            "flagged_fraud": int(predictions["is_fraud"].sum()),
            "fraud_rate_pct": round(predictions["is_fraud"].mean() * 100, 2),
        }

        unique = np.unique(labels)
        if len(unique) > 1:
            metrics["silhouette_score"] = round(
                float(silhouette_score(X_scaled, labels)), 4
            )
            metrics["calinski_harabasz_score"] = round(
                float(calinski_harabasz_score(X_scaled, labels)), 4
            )

        # Statistical comparison
        fraud_mask = predictions["is_fraud"] == 1
        if fraud_mask.any() and (~fraud_mask).any():
            f_stats = predictions.loc[fraud_mask, "total_amount"].describe()
            n_stats = predictions.loc[~fraud_mask, "total_amount"].describe()
            metrics["avg_fraud_amount"] = round(float(f_stats["mean"]), 2)
            metrics["avg_normal_amount"] = round(float(n_stats["mean"]), 2)
            metrics["fraud_amount_std"] = round(float(f_stats["std"]), 2)
            metrics["normal_amount_std"] = round(float(n_stats["std"]), 2)

        metrics["avg_anomaly_score"] = round(
            float(predictions["anomaly_score"].mean()), 4
        )
        metrics["anomaly_score_std"] = round(
            float(predictions["anomaly_score"].std()), 4
        )

        logger.info(f"Fraud evaluation: {metrics}")
        return metrics

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                },
                f,
            )
        logger.info(f"FraudDetector saved → {path}")

    # ── Supervised Evaluation ──────────────────────────────────────────────

    def evaluate_supervised(
        self,
        predictions: pd.DataFrame,
        true_labels: pd.Series,
    ) -> Dict:
        """Compute precision, recall, F1, accuracy against known labels."""
        y_true = true_labels.values
        y_pred = predictions["is_fraud"].values

        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Fraud"],
            zero_division=0,
        )
        logger.info(f"Supervised evaluation:\n{report}")
        logger.info(f"Metrics: {metrics}")
        return metrics

    @classmethod
    def load(cls, path: Path) -> "FraudDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        logger.info(f"FraudDetector loaded ← {path}")
        return obj


# =====================================================================
#  Supervised Fraud Detector (Random Forest + class balancing)
# =====================================================================

class SupervisedFraudDetector:
    """Supervised fraud classification when ground-truth labels exist."""

    def __init__(self) -> None:
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.threshold: float = 0.5  # probability threshold (tuned during fit)

    # ── Feature Engineering (shared with unsupervised) ──────────────────

    engineer_features = staticmethod(FraudDetector.engineer_features)

    # ── Threshold Tuning ────────────────────────────────────────────────

    @staticmethod
    def _find_optimal_threshold(
        model, X_scaled: np.ndarray, y: np.ndarray
    ) -> float:
        """Search probability thresholds 0.1–0.9 to maximise F1 on OOB / CV."""
        from sklearn.metrics import f1_score as _f1
        probas = model.predict_proba(X_scaled)[:, 1]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.10, 0.91, 0.05):
            preds = (probas >= t).astype(int)
            f = float(_f1(y, preds, zero_division=0))
            if f > best_f1:
                best_f1, best_t = f, t
        return round(float(best_t), 2)

    # ── Training ────────────────────────────────────────────────────────

    def fit(
        self,
        claim_features: pd.DataFrame,
        labels: pd.Series,
    ) -> "SupervisedFraudDetector":
        """Train Random Forest with optional GridSearchCV fine-tuning
        and RepeatedStratifiedKFold cross-validation."""
        self.feature_names = [
            c for c in claim_features.columns if c not in (cfg.CLAIM_ID_COL, "IS_FRAUD")
        ]
        X = claim_features[self.feature_names].values
        y = labels.values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # ── Cross-validation strategy ──────────────────────────────
        n_splits = getattr(cfg, "CV_N_SPLITS", 5)
        n_repeats = getattr(cfg, "CV_N_REPEATS", 3)
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats,
            random_state=cfg.RANDOM_STATE,
        )

        # ── Hyperparameter fine-tuning via GridSearchCV ────────────
        grid_enabled = getattr(cfg, "GRID_SEARCH_ENABLED", False)
        if grid_enabled:
            param_grid = getattr(cfg, "RF_PARAM_GRID", {})
            logger.info(
                f"Running GridSearchCV over {len(param_grid)} params "
                f"({n_splits}-fold × {n_repeats} repeats) ..."
            )
            base_rf = RandomForestClassifier(
                class_weight="balanced",
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
            )
            grid = GridSearchCV(
                base_rf,
                param_grid,
                cv=rskf,
                scoring="f1",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_scaled, y)
            self.model = grid.best_estimator_
            self._grid_best_params = grid.best_params_
            self._grid_best_score = round(float(grid.best_score_), 4)
            logger.info(f"  GridSearch best params: {grid.best_params_}")
            logger.info(f"  GridSearch best CV-F1 : {grid.best_score_:.4f}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=cfg.RF_N_ESTIMATORS,
                max_depth=cfg.RF_MAX_DEPTH,
                min_samples_split=cfg.RF_MIN_SAMPLES_SPLIT,
                class_weight="balanced",
                random_state=cfg.RANDOM_STATE,
                n_jobs=-1,
            )
            self.model.fit(X_scaled, y)

        # ── Optimal probability threshold ────────────────────────
        self.threshold = self._find_optimal_threshold(self.model, X_scaled, y)

        # ── RepeatedStratifiedKFold cross-validation (final model) ─
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_results = cross_validate(
            self.model, X_scaled, y, cv=rskf,
            scoring=scoring, return_train_score=False,
        )

        self._cv_metrics = {}
        self._cv_fold_details = {}
        for metric in scoring:
            arr = cv_results[f"test_{metric}"]
            self._cv_metrics[f"cv_{metric}"] = (
                f"{arr.mean():.4f} +/- {arr.std():.4f}"
            )
            self._cv_fold_details[f"cv_{metric}_folds"] = [
                round(float(v), 4) for v in arr
            ]

        self._cv_metrics["cv_n_splits"] = n_splits
        self._cv_metrics["cv_n_repeats"] = n_repeats
        self._cv_metrics["cv_total_folds"] = n_splits * n_repeats

        logger.info(
            f"Supervised fraud detector trained on {X.shape[0]} claims × "
            f"{X.shape[1]} features  |  Threshold: {self.threshold}"
        )
        logger.info(f"  Repeated Stratified K-Fold: {n_splits} folds × {n_repeats} repeats ")
        logger.info(f"  CV-Accuracy : {self._cv_metrics['cv_accuracy']}")
        logger.info(f"  CV-Precision: {self._cv_metrics['cv_precision']}")
        logger.info(f"  CV-Recall   : {self._cv_metrics['cv_recall']}")
        logger.info(f"  CV-F1       : {self._cv_metrics['cv_f1']}")
        logger.info(f"  CV-ROC-AUC  : {self._cv_metrics['cv_roc_auc']}")

        if grid_enabled:
            logger.info(f"  Best Params : {self._grid_best_params}")

        return self

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(self, claim_features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not fitted – call fit() first")

        feat_cols = [c for c in self.feature_names if c in claim_features.columns]
        X = claim_features[feat_cols].values
        X_scaled = self.scaler.transform(X)

        results = claim_features.copy()
        results["fraud_probability"] = self.model.predict_proba(X_scaled)[:, 1]
        results["is_fraud"] = (results["fraud_probability"] >= self.threshold).astype(int)

        n_fraud = int(results["is_fraud"].sum())
        logger.info(
            f"Supervised predicted: {n_fraud} suspicious claims "
            f"({n_fraud / len(results) * 100:.1f}%) [threshold={self.threshold}]"
        )
        return results

    # ── Evaluation ──────────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: pd.DataFrame,
        true_labels: pd.Series,
    ) -> Dict:
        """Full evaluation: accuracy, precision, recall, F1, ROC-AUC."""
        y_true = true_labels.values
        y_pred = predictions["is_fraud"].values

        metrics: Dict = {
            "threshold_used": self.threshold,
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

        # ROC-AUC (requires probability column)
        if "fraud_probability" in predictions.columns:
            try:
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_true, predictions["fraud_probability"].values)), 4
                )
            except ValueError:
                pass  # only one class in y_true

        # Cross-validation metrics (from training phase)
        if hasattr(self, "_cv_metrics"):
            metrics["cross_validation"] = self._cv_metrics

        # Per-fold details (attached if available)
        if hasattr(self, "_cv_fold_details"):
            metrics["cross_validation_fold_details"] = self._cv_fold_details

        # Grid search best params (if tuning was performed)
        if hasattr(self, "_grid_best_params"):
            metrics["grid_search"] = {
                "best_params": self._grid_best_params,
                "best_cv_f1": self._grid_best_score,
            }

        # Feature importances
        if self.model is not None:
            importances = dict(
                zip(self.feature_names, self.model.feature_importances_.round(4))
            )
            metrics["feature_importances"] = dict(
                sorted(importances.items(), key=lambda x: x[1], reverse=True)
            )

        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Fraud"],
            zero_division=0,
        )
        logger.info(f"Supervised evaluation:\n{report}")
        logger.info(f"Metrics: {metrics}")
        return metrics

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                    "cv_metrics": getattr(self, "_cv_metrics", {}),
                    "cv_fold_details": getattr(self, "_cv_fold_details", {}),
                    "grid_best_params": getattr(self, "_grid_best_params", None),
                },
                f,
            )
        logger.info(f"SupervisedFraudDetector saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "SupervisedFraudDetector":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.threshold = data.get("threshold", 0.5)
        obj._cv_metrics = data.get("cv_metrics", {})
        obj._cv_fold_details = data.get("cv_fold_details", {})
        if data.get("grid_best_params"):
            obj._grid_best_params = data["grid_best_params"]
        logger.info(f"SupervisedFraudDetector loaded ← {path}")
        return obj
