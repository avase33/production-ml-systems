"""
Fraud Detection Model Training with SHAP Explainability
========================================================
Trains an XGBoost fraud classifier, evaluates it, and generates SHAP
explanations for regulatory audit compliance (OCC SR 11-7).

Author: Akhil Vase | Senior AI/ML Engineer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "tx_count_1h", "tx_count_24h", "amount_sum_1h", "amount_sum_24h",
    "avg_amount_30d", "amount_zscore", "hour_of_day", "is_weekend",
    "time_since_last_tx_secs", "is_cross_border",
    "high_risk_merchant_category", "amount_log", "device_seen_before",
]
LABEL_COL = "is_fraud"

XGBOOST_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "logloss"],
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "scale_pos_weight": 50,   # handles class imbalance (~2% fraud rate)
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class FraudModelTrainer:
    """
    Trains, evaluates, and registers a fraud detection XGBoost model.

    Usage
    -----
    trainer = FraudModelTrainer(mlflow_experiment="fraud-detection-v3")
    model, metrics = trainer.train(df)
    trainer.explain(model, X_test)
    """

    def __init__(
        self,
        mlflow_experiment: str = "fraud-detection",
        model_output_dir: str = "models/fraud",
        precision_target: float = 0.90,
    ):
        self.experiment = mlflow_experiment
        self.output_dir = Path(model_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.precision_target = precision_target
        mlflow.set_experiment(mlflow_experiment)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self, df: pd.DataFrame
    ) -> tuple[xgb.XGBClassifier, dict[str, float]]:
        """Full training run: split, train, evaluate, log to MLflow, return model + metrics."""
        X, y = df[FEATURE_COLS], df[LABEL_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        log.info("Training set: %d rows | Fraud rate: %.2f%%",
                 len(X_train), y_train.mean() * 100)

        with mlflow.start_run():
            mlflow.log_params(XGBOOST_PARAMS)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("fraud_rate_pct", round(y_train.mean() * 100, 3))

            model = self._fit(X_train, y_train, X_test, y_test)
            metrics = self._evaluate(model, X_test, y_test)

            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, artifact_path="fraud_model")

            # Log feature importance
            fi_path = self.output_dir / "feature_importance.json"
            fi_path.write_text(
                json.dumps(dict(zip(FEATURE_COLS, model.feature_importances_.tolist())), indent=2)
            )
            mlflow.log_artifact(str(fi_path))

            log.info("Metrics: %s", metrics)
            run_id = mlflow.active_run().info.run_id
            log.info("MLflow run: %s", run_id)

        return model, metrics

    def explain(
        self,
        model: xgb.XGBClassifier,
        X: pd.DataFrame,
        n_samples: int = 500,
    ) -> np.ndarray:
        """
        Generate SHAP values for audit / regulatory explainability.
        Saves summary plot and per-feature mean |SHAP| to output dir.
        """
        log.info("Computing SHAP values on %d samples …", min(n_samples, len(X)))
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        mean_abs_shap = pd.Series(
            np.abs(shap_values).mean(axis=0), index=FEATURE_COLS
        ).sort_values(ascending=False)

        shap_path = self.output_dir / "shap_importance.json"
        shap_path.write_text(mean_abs_shap.to_json(indent=2))
        log.info("SHAP feature importance:\n%s", mean_abs_shap.to_string())

        return shap_values

    def explain_single(
        self,
        model: xgb.XGBClassifier,
        feature_vector: np.ndarray,
    ) -> dict[str, float]:
        """
        Explain a single prediction — used in production for audit trail logging.
        Returns {feature_name: shap_value} for the flagged transaction.
        """
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(feature_vector.reshape(1, -1))[0]
        return dict(zip(FEATURE_COLS, sv.tolist()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=100,
        )
        return model

    def _evaluate(
        self,
        model: xgb.XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        proba = model.predict_proba(X_test)[:, 1]

        # Find threshold that hits precision target
        precision, recall, thresholds = precision_recall_curve(y_test, proba)
        valid = precision >= self.precision_target
        if valid.any():
            best_idx = np.argmax(recall[valid])
            threshold = thresholds[np.where(valid)[0][best_idx]]
        else:
            threshold = 0.5
            log.warning("Precision target %.2f not achievable; using threshold=0.5",
                        self.precision_target)

        preds = (proba >= threshold).astype(int)
        report = classification_report(y_test, preds, output_dict=True)

        return {
            "auc_roc": round(roc_auc_score(y_test, proba), 4),
            "precision": round(report["1"]["precision"], 4),
            "recall": round(report["1"]["recall"], 4),
            "f1": round(report["1"]["f1-score"], 4),
            "threshold": round(float(threshold), 4),
        }


# ---------------------------------------------------------------------------
# Cross-validation for champion-challenger comparison
# ---------------------------------------------------------------------------

def cross_validate_model(df: pd.DataFrame, n_splits: int = 5) -> dict[str, float]:
    """
    Stratified k-fold CV — used before promoting a challenger model to champion.
    Returns mean ± std of key metrics.
    """
    X, y = df[FEATURE_COLS], df[LABEL_COL]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, precisions, recalls = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=20, verbose=False)

        proba = model.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        report = classification_report(y_val, preds, output_dict=True)

        aucs.append(roc_auc_score(y_val, proba))
        precisions.append(report["1"]["precision"])
        recalls.append(report["1"]["recall"])
        log.info("Fold %d → AUC=%.4f  Precision=%.4f  Recall=%.4f",
                 fold, aucs[-1], precisions[-1], recalls[-1])

    return {
        "cv_auc_mean": round(np.mean(aucs), 4),
        "cv_auc_std": round(np.std(aucs), 4),
        "cv_precision_mean": round(np.mean(precisions), 4),
        "cv_recall_mean": round(np.mean(recalls), 4),
    }
