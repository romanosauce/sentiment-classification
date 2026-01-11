"""
Evaluation script for sentiment classifier (for DVC pipeline).

Usage:
    python -m src.evaluate --model-path models/sentiment_model --config configs/train_config.yaml
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.dataset import SentimentDataset
from src.model import SentimentClassifier
from src.preprocessing import TextPreprocessor
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument("--model-path", type=str, required=True, help="Model directory")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--prepared-data-dir", type=str, default="processed", help="Prepared data dir")
    parser.add_argument("--output", type=str, default="metrics.json", help="Output metrics file")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def evaluate_model(model, data_loader, device) -> dict:
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch["input_ids"].to(device))
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds, all_labels, all_probs = map(np.array, [all_preds, all_labels, all_probs])

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_score": float(f1_score(all_labels, all_preds, average="weighted")),
        "precision": float(precision_score(all_labels, all_preds, average="weighted")),
        "recall": float(recall_score(all_labels, all_preds, average="weighted")),
        "roc_auc": float(roc_auc_score(all_labels, all_probs)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {args.model_path}")
    model = SentimentClassifier.from_pretrained(args.model_path, device)
    preprocessor = TextPreprocessor.load(args.model_path)

    with open(Path(args.prepared_data_dir) / "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    test_dataset = SentimentDataset(test_data["texts"], test_data["labels"], preprocessor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    logger.info("Evaluating...")
    metrics = evaluate_model(model, test_loader, device)

    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
