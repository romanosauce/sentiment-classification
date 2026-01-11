"""
Export trained model to TorchScript format for TorchServe.

Usage:
    python serve/export_model.py
    python serve/export_model.py --model_path models/sentiment_model --output_path serve/model.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SentimentClassifier
from src.preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to TorchScript")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/sentiment_model",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="serve/model.pt",
        help="Output path for TorchScript model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading model from {args.model_path}")
    model = SentimentClassifier.from_pretrained(args.model_path, device=torch.device("cpu"))
    model.eval()

    # Load preprocessor config for index_to_name mapping
    preprocessor = TextPreprocessor.load(args.model_path)

    # Create example input for tracing
    batch_size = 1
    seq_length = preprocessor.max_seq_length
    example_input = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.long)

    logger.info("Tracing model to TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Save TorchScript model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    logger.info(f"TorchScript model saved to {output_path}")

    # Create index_to_name.json for TorchServe
    index_to_name = {
        "0": "negative",
        "1": "positive"
    }
    index_to_name_path = output_path.parent / "index_to_name.json"
    with open(index_to_name_path, "w") as f:
        json.dump(index_to_name, f, indent=2)
    logger.info(f"Index to name mapping saved to {index_to_name_path}")

    # Copy preprocessor config for handler
    preprocessor_src = Path(args.model_path) / "preprocessor.json"
    preprocessor_dst = output_path.parent / "preprocessor.json"
    if preprocessor_src.exists():
        import shutil
        shutil.copy(preprocessor_src, preprocessor_dst)
        logger.info(f"Preprocessor config copied to {preprocessor_dst}")

    # Verify the exported model
    logger.info("Verifying exported model...")
    loaded_model = torch.jit.load(str(output_path))
    with torch.no_grad():
        output = loaded_model(example_input)
    logger.info(f"Model output shape: {output.shape}")
    logger.info("Export completed successfully!")


if __name__ == "__main__":
    main()
