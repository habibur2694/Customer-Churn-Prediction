"""
predictor.py
============
Loads the trained model and provides a simple interface for predicting
churn probability for a single customer record.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from tensorflow import keras

from src.preprocessing import prepare_single_record
from src.utils import MODEL_PATH, FeatureConfig, get_logger

logger = get_logger(__name__)

_model_cache: keras.Model | None = None


def load_trained_model(model_path: str = MODEL_PATH) -> keras.Model:
    """
    Load the trained Keras model from disk, caching it in memory so
    repeated Streamlit reruns do not reload the model from disk.

    Args:
        model_path: Path to the saved ``.keras`` model file.

    Returns:
        The loaded Keras model.
    """
    global _model_cache
    if _model_cache is None:
        logger.info("Loading trained model from %s", model_path)
        _model_cache = keras.models.load_model(model_path)
    return _model_cache


def predict_customer(record: Dict, config: FeatureConfig | None = None) -> Tuple[str, float, float]:
    """
    Predict whether a single customer will stay or churn.

    Args:
        record: Dictionary of raw customer feature values, matching the
            schema defined in ``FeatureConfig``.
        config: Optional feature configuration; defaults to standard schema.

    Returns:
        Tuple of (label, confidence, raw_churn_probability) where label
        is "Stay" or "Churn", confidence is the model's probability for
        that label (0-1), and raw_churn_probability is always P(Churn).
    """
    config = config or FeatureConfig()
    model = load_trained_model()

    X = prepare_single_record(record, config)
    churn_probability = float(model.predict(X, verbose=0).ravel()[0])

    if churn_probability >= 0.5:
        label = "Churn"
        confidence = churn_probability
    else:
        label = "Stay"
        confidence = 1.0 - churn_probability

    logger.info("Prediction: %s (churn_probability=%.4f, confidence=%.4f)", label, churn_probability, confidence)
    return label, confidence, churn_probability
