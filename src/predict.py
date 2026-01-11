"""
Batch prediction script for sentiment classifier.

Usage:
    python -m src.predict --input_path data.csv --output_path preds.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.inference import SentimentPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch sentiment prediction")
    parser.add_argument("--input_path", type=str, required=True, help="Input CSV with 'text' column")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV path")
    parser.add_argument("--model_path", type=str, default="models/sentiment_model", help="Model directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SentimentPredictor.from_pretrained(args.model_path, device)

    logger.info(f"Loading data from {args.input_path}")
    df = pd.read_csv(args.input_path)
    texts = df[args.text_column].tolist()

    logger.info(f"Predicting {len(texts)} samples...")
    results = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Predicting"):
        batch = texts[i:i + args.batch_size]
        for text in batch:
            if pd.isna(text) or not str(text).strip():
                results.append({"prediction": "unknown", "confidence": 0.0,
                               "prob_positive": 0.0, "prob_negative": 0.0})
            else:
                pred = predictor.predict(str(text))
                results.append({
                    "prediction": pred.sentiment,
                    "confidence": pred.confidence,
                    "prob_positive": pred.probabilities.get("positive", 0.0),
                    "prob_negative": pred.probabilities.get("negative", 0.0),
                })

    output_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_path, index=False)

    positive = sum(1 for r in results if r["prediction"] == "positive")
    negative = sum(1 for r in results if r["prediction"] == "negative")
    logger.info(f"Done! Positive: {positive}, Negative: {negative}")
    logger.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
