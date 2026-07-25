"""
trainer.py
==========
Handles model training with EarlyStopping, ModelCheckpoint, and a
validation split. Saves the trained model and training history.
"""

from __future__ import annotations

import pickle
from typing import Tuple

import numpy as np
from tensorflow import keras

from src.model import build_model
from src.utils import HISTORY_PATH, MODEL_PATH, get_logger

logger = get_logger(__name__)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    patience: int = 10,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train the churn prediction model with early stopping and
    checkpointing on the best validation loss.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        validation_split: Fraction of training data used for validation.
        patience: Number of epochs with no improvement before stopping.

    Returns:
        Tuple of (trained model, training history).
    """
    model = build_model(input_dim=X_train.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logger.info(
        "Starting training: epochs=%d, batch_size=%d, validation_split=%.2f",
        epochs,
        batch_size,
        validation_split,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=2,
    )

    # ModelCheckpoint already saved the best model, but ensure the final
    # (best-weights-restored) model is persisted as well.
    model.save(MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)
    logger.info("Training history saved to %s", HISTORY_PATH)

    return model, history
