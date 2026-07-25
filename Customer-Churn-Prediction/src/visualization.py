"""
visualization.py
=================
Generates and saves Matplotlib charts: training accuracy/loss curves,
confusion matrix heatmap, and ROC curve. Charts are saved to the
``assets`` directory so the Streamlit app can display cached images
in addition to being able to plot live.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # Safe for headless/server-side rendering.
import matplotlib.pyplot as plt
import numpy as np

from src.utils import ASSETS_DIR, get_logger

logger = get_logger(__name__)

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "ggplot")


def plot_training_history(history: Dict, save: bool = True):
    """
    Plot training/validation accuracy and loss curves.

    Args:
        history: Dictionary produced by ``keras.callbacks.History.history``.
        save: Whether to save the figures to the assets directory.

    Returns:
        Tuple of (accuracy_figure, loss_figure).
    """
    # Accuracy plot
    fig_acc, ax_acc = plt.subplots(figsize=(7, 5))
    ax_acc.plot(history.get("accuracy", []), label="Train Accuracy", linewidth=2)
    ax_acc.plot(history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    ax_acc.set_title("Model Accuracy over Epochs")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    fig_acc.tight_layout()

    # Loss plot
    fig_loss, ax_loss = plt.subplots(figsize=(7, 5))
    ax_loss.plot(history.get("loss", []), label="Train Loss", linewidth=2)
    ax_loss.plot(history.get("val_loss", []), label="Validation Loss", linewidth=2)
    ax_loss.set_title("Model Loss over Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    fig_loss.tight_layout()

    if save:
        fig_acc.savefig(os.path.join(ASSETS_DIR, "training_accuracy.png"), dpi=150)
        fig_loss.savefig(os.path.join(ASSETS_DIR, "training_loss.png"), dpi=150)
        logger.info("Saved training history charts to %s", ASSETS_DIR)

    return fig_acc, fig_loss


def plot_confusion_matrix(cm: np.ndarray, save: bool = True):
    """
    Plot a labeled confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array.
        save: Whether to save the figure to the assets directory.

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Stay", "Churn"]

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"), dpi=150)
        logger.info("Saved confusion matrix chart to %s", ASSETS_DIR)

    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save: bool = True):
    """
    Plot the ROC curve with the AUC score annotated.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        auc: Area under the ROC curve.
        save: Whether to save the figure to the assets directory.

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(ASSETS_DIR, "roc_curve.png"), dpi=150)
        logger.info("Saved ROC curve chart to %s", ASSETS_DIR)

    return fig


def plot_prediction_probability(probability: float):
    """
    Plot a horizontal bar gauge showing the churn probability for a
    single customer prediction, used live in the Streamlit app.

    Args:
        probability: Predicted probability of churn (0.0 - 1.0).

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 1.4))
    color = "#e74c3c" if probability >= 0.5 else "#2ecc71"

    ax.barh([0], [1], color="#2b2b3a", height=0.5)
    ax.barh([0], [probability], color=color, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(f"Churn Probability: {probability * 100:.1f}%", fontsize=12, fontweight="bold")
    fig.tight_layout()

    return fig
