from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_dataset
from src.evaluator import evaluate_model
from src.preprocessing import prepare_dataset
from src.trainer import train_model
from src.utils import FeatureConfig, ensure_directories, get_logger
from src.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_history

logger = get_logger(__name__)


def main() -> None:
    """Run the full training and evaluation pipeline."""
    ensure_directories()
    config = FeatureConfig()

    logger.info("Step 1/4: Loading dataset ...")
    df = load_dataset(n_records=10_000)

    logger.info("Step 2/4: Preprocessing dataset ...")
    X_train, X_test, y_train, y_test, feature_columns = prepare_dataset(df, config)
    logger.info("Feature columns (%d): %s", len(feature_columns), feature_columns)

    logger.info("Step 3/4: Training model ...")
    model, history = train_model(X_train, y_train, epochs=100, batch_size=32, patience=10)

    logger.info("Step 4/4: Evaluating model ...")
    metrics = evaluate_model(model, X_test, y_test)

    plot_training_history(history.history)
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"])

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
    print("=" * 60)
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
