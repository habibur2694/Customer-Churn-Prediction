"""
evaluator.py
============
Computes evaluation metrics (accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, classification report) for the trained churn model.
"""

from __future__ import annotations

import pickle
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

from src.utils import METRICS_PATH, get_logger

logger = get_logger(__name__)


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate the trained model on the held-out test set and compute a
    comprehensive set of classification metrics.

    Args:
        model: Trained Keras model.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dictionary containing all computed metrics and raw arrays
        needed for downstream plotting (confusion matrix, ROC curve).
    """
    y_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Stay", "Churn"], zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "classification_report": report,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    logger.info(
        "Evaluation complete -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        accuracy,
        precision,
        recall,
        f1,
        auc,
    )
    logger.info("Classification report:\n%s", report)

    return metrics
