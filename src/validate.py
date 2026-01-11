"""
Validation script for sentiment classifier.

Usage:
    python -m src.validate --model-path models/sentiment_model --config configs/train_config.yaml
"""

import argparse
import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from src.data_loader import load_imdb_data
from src.dataset import SentimentDataset
from src.inference import SentimentPredictor
from src.model import SentimentClassifier
from src.preprocessing import TextPreprocessor
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate sentiment classifier")
    parser.add_argument("--model-path", type=str, required=True, help="Model directory")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data-path", type=str, default=None, help="Test data path")
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

    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {args.model_path}")
    model = SentimentClassifier.from_pretrained(args.model_path, device)
    preprocessor = TextPreprocessor.load(args.model_path)

    data_path = args.data_path or config["data"]["test_path"]
    test_texts, test_labels = load_imdb_data(data_path)
    logger.info(f"Test samples: {len(test_texts)}")

    test_dataset = SentimentDataset(test_texts, test_labels, preprocessor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    results = evaluate_model(model, test_loader, device)

    print(f"\nAccuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}, ROC-AUC: {results['roc_auc']:.4f}")
    print("\n" + classification_report(results["labels"], results["preds"], target_names=["Negative", "Positive"]))

    # Sample predictions
    predictor = SentimentPredictor(model, preprocessor, device)
    samples = ["This movie was fantastic!", "Terrible film, waste of time.", "It was okay, nothing special."]
    print("\nSample predictions:")
    for text in samples:
        r = predictor.predict(text)
        print(f"  '{text[:40]}...' -> {r.sentiment} ({r.confidence:.2f})")


if __name__ == "__main__":
    main()
