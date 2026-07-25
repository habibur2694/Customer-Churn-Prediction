"""
utils.py
========
Shared utilities for logging, path management, and configuration
constants used across the Customer Churn Prediction System.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
ASSETS_DIR: str = os.path.join(BASE_DIR, "assets")

DATA_PATH: str = os.path.join(DATA_DIR, "customer_churn.csv")
MODEL_PATH: str = os.path.join(MODELS_DIR, "churn_model.keras")
SCALER_PATH: str = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODERS_PATH: str = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURE_COLUMNS_PATH: str = os.path.join(MODELS_DIR, "feature_columns.pkl")
HISTORY_PATH: str = os.path.join(MODELS_DIR, "history.pkl")
METRICS_PATH: str = os.path.join(MODELS_DIR, "metrics.pkl")

TARGET_COLUMN: str = "Churn"
RANDOM_STATE: int = 42


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, ASSETS_DIR):
        os.makedirs(directory, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger with console output.

    Args:
        name: Name of the logger, typically ``__name__`` of the caller.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@dataclass
class FeatureConfig:
    """Configuration describing the dataset schema."""

    numerical_features: List[str] = field(
        default_factory=lambda: [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Age",
            "NumServices",
        ]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "TechSupport",
            "StreamingTV",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
    )
    target: str = TARGET_COLUMN
