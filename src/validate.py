"""
Validation script for sentiment classifier.

Usage:
    python src/validate.py --model-path models/sentiment_model --config configs/train_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_imdb_data
from src.dataset import SentimentDataset
from src.inference import SentimentPredictor
from src.model import SentimentClassifier
from src.preprocessing import TextPreprocessor
from src.utils import load_config, setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate sentiment classifier")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to test data (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def evaluate_model(
    model: SentimentClassifier,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model and compute metrics.
    
    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return {
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted"),
        "roc_auc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=["negative", "positive"]
        ),
    }


def print_results(results: dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nAccuracy:  {results['accuracy']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    
    print("\nClassification Report:")
    print(classification_report(
        results["labels"],
        results["predictions"],
        target_names=["Negative", "Positive"],
    ))
    
    print("=" * 60)


def test_sample_predictions(predictor: SentimentPredictor) -> None:
    """Test model with sample texts."""
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    test_texts = [
        "This movie was absolutely fantastic! Great acting and storyline.",
        "Terrible film. Complete waste of time and money.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've ever seen. Highly recommended!",
        "Boring and predictable. I fell asleep halfway through.",
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text[:60]}...")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Confidence: {result.confidence:.4f} ({result.confidence_level})")
        print(f"  Inference time: {result.inference_time_ms:.2f} ms")


def main() -> None:
    """Main validation function."""
    args = parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info(f"Loading model from {args.model_path}...")
    model = SentimentClassifier.from_pretrained(args.model_path, device)
    preprocessor = TextPreprocessor.load(args.model_path)
    
    data_path = args.data_path or config["data"]["test_path"]
    logger.info(f"Loading test data from {data_path}...")
    test_texts, test_labels = load_imdb_data(data_path)
    
    logger.info(f"Test samples: {len(test_texts)}")
    
    test_dataset = SentimentDataset(test_texts, test_labels, preprocessor)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    logger.info("Running evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    print_results(results)
    
    predictor = SentimentPredictor(model, preprocessor, device)
    test_sample_predictions(predictor)


if __name__ == "__main__":
    main()
