"""
model.py
========
Defines the TensorFlow/Keras Sequential neural network architecture
used for binary customer churn classification.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import get_logger

logger = get_logger(__name__)


def build_model(input_dim: int, learning_rate: float = 0.001) -> keras.Model:
    """
    Build and compile the churn prediction neural network.

    Architecture:
        Input -> Dense(128, ReLU) -> BatchNorm -> Dropout
              -> Dense(64, ReLU) -> Dropout
              -> Dense(32, ReLU)
              -> Dense(1, Sigmoid)

    Args:
        input_dim: Number of input features.
        learning_rate: Learning rate for the Adam optimizer.

    Returns:
        A compiled ``keras.Model`` ready for training.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_layer"),
            layers.Dense(128, activation="relu", name="dense_1"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Dropout(0.3, name="dropout_1"),
            layers.Dense(64, activation="relu", name="dense_2"),
            layers.Dropout(0.2, name="dropout_2"),
            layers.Dense(32, activation="relu", name="dense_3"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="churn_prediction_model",
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info("Built model with input_dim=%d and %d total parameters.", input_dim, model.count_params())
    return model


if __name__ == "__main__":
    tf.random.set_seed(42)
    m = build_model(input_dim=18)
    m.summary()
