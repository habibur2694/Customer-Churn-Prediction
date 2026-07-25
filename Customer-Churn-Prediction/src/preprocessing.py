"""
preprocessing.py
================
Data cleaning, encoding, scaling, and train/test splitting utilities
for the Customer Churn Prediction System.
"""

from __future__ import annotations

import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import (
    ENCODERS_PATH,
    FEATURE_COLUMNS_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    FeatureConfig,
    get_logger,
)

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset by removing duplicates, dropping identifier
    columns, and handling missing values.

    Args:
        df: Raw input DataFrame.

    Returns:
        A cleaned copy of the DataFrame.
    """
    df = df.copy()

    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info("Removed %d duplicate rows.", before - len(df))

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Coerce numeric-looking object columns (e.g. TotalCharges with blanks).
    for col in ("TotalCharges", "MonthlyCharges", "tenure", "Age", "NumServices"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled %d missing values in '%s' with median %.2f.", df[col].isna().sum(), col, median_val)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Filled missing values in '%s' with mode '%s'.", col, mode_val)

    df.reset_index(drop=True, inplace=True)
    return df


def encode_features(
    df: pd.DataFrame, config: FeatureConfig, fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical columns using per-column label encoders, and
    encode the binary target column as 0/1.

    Args:
        df: Cleaned DataFrame.
        config: Feature configuration describing column roles.
        fit: If True, fit new encoders; otherwise load saved encoders.

    Returns:
        Tuple of (encoded DataFrame, dict of fitted encoders).
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    if fit:
        for col in config.categorical_features:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder

        if config.target in df.columns:
            target_encoder = LabelEncoder()
            df[config.target] = target_encoder.fit_transform(df[config.target].astype(str))
            encoders[config.target] = target_encoder

        with open(ENCODERS_PATH, "wb") as f:
            pickle.dump(encoders, f)
        logger.info("Fitted and saved %d encoders to %s", len(encoders), ENCODERS_PATH)
    else:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
        for col in config.categorical_features:
            encoder = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda v, enc=encoder: v if v in enc.classes_ else enc.classes_[0]
            )
            df[col] = encoder.transform(df[col])

    return df, encoders


def scale_features(
    df: pd.DataFrame, config: FeatureConfig, fit: bool = True
) -> pd.DataFrame:
    """
    Standardize numerical features to zero mean and unit variance.

    Args:
        df: Encoded DataFrame.
        config: Feature configuration describing column roles.
        fit: If True, fit a new scaler; otherwise load the saved scaler.

    Returns:
        DataFrame with scaled numerical columns.
    """
    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[config.numerical_features] = scaler.fit_transform(df[config.numerical_features])
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Fitted and saved scaler to %s", SCALER_PATH)
    else:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        df[config.numerical_features] = scaler.transform(df[config.numerical_features])

    return df


def prepare_dataset(
    df: pd.DataFrame, config: FeatureConfig, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Run the full preprocessing pipeline: clean, encode, scale, and split.

    Args:
        df: Raw input DataFrame.
        config: Feature configuration describing column roles.
        test_size: Fraction of data reserved for the test set.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_column_order).
    """
    df = clean_data(df)
    df, _ = encode_features(df, config, fit=True)
    df = scale_features(df, config, fit=True)

    feature_columns = config.numerical_features + config.categorical_features
    with open(FEATURE_COLUMNS_PATH, "wb") as f:
        pickle.dump(feature_columns, f)

    X = df[feature_columns].values.astype(np.float32)
    y = df[config.target].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Split dataset -> train: %s, test: %s (positive rate train=%.3f, test=%.3f)",
        X_train.shape,
        X_test.shape,
        y_train.mean(),
        y_test.mean(),
    )
    return X_train, X_test, y_train, y_test, feature_columns


def prepare_single_record(record: Dict, config: FeatureConfig) -> np.ndarray:
    """
    Preprocess a single customer record (from the Streamlit form) into
    a model-ready feature vector using the saved encoders and scaler.

    Args:
        record: Dictionary mapping feature names to raw values.
        config: Feature configuration describing column roles.

    Returns:
        A 2D numpy array of shape (1, n_features) ready for prediction.
    """
    df = pd.DataFrame([record])
    df, _ = encode_features(df, config, fit=False)
    df = scale_features(df, config, fit=False)

    with open(FEATURE_COLUMNS_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    return df[feature_columns].values.astype(np.float32)
