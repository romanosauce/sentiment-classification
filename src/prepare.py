"""
Data preparation script for sentiment classifier.

Converts raw IMDB data from individual files to a processed format
suitable for training.

Usage:
    python -m src.prepare --config configs/train_config.yaml
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    get_data_statistics,
    load_imdb_data,
    print_data_statistics,
    split_data,
)
from src.preprocessing import TextPreprocessor
from src.utils import load_config, set_seed, setup_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare data for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/prepared",
        help="Output directory for prepared data",
    )

    return parser.parse_args()


def main() -> None:
    """Main data preparation function."""
    args = parse_args()

    setup_logging(log_level="INFO")

    config = load_config(args.config)
    seed = config["training"]["random_seed"]
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting data preparation...")
    logger.info("=" * 60)

    logger.info("Loading training data...")
    train_texts, train_labels = load_imdb_data(config["data"]["train_path"])

    stats = get_data_statistics(train_texts, train_labels)
    print_data_statistics(stats)

    train_texts, train_labels, val_texts, val_labels = split_data(
        train_texts,
        train_labels,
        validation_split=config["training"]["validation_split"],
        random_seed=seed,
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    logger.info("Loading test data...")
    test_texts, test_labels = load_imdb_data(config["data"]["test_path"])
    logger.info(f"Test samples: {len(test_texts)}")

    logger.info("Building vocabulary...")
    preprocessor = TextPreprocessor(
        max_vocab_size=config["data"]["max_vocab_size"],
        max_seq_length=config["data"]["max_seq_length"],
        min_word_freq=config["data"]["min_word_freq"],
    )
    preprocessor.fit(train_texts)

    vocab_size = preprocessor.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")

    logger.info("Saving prepared data...")

    with open(output_dir / "train_data.pkl", "wb") as f:
        pickle.dump({
            "texts": train_texts,
            "labels": train_labels,
        }, f)

    with open(output_dir / "val_data.pkl", "wb") as f:
        pickle.dump({
            "texts": val_texts,
            "labels": val_labels,
        }, f)

    with open(output_dir / "test_data.pkl", "wb") as f:
        pickle.dump({
            "texts": test_texts,
            "labels": test_labels,
        }, f)

    preprocessor.save(str(output_dir))

    metadata = {
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "test_samples": len(test_texts),
        "vocab_size": vocab_size,
        "max_seq_length": config["data"]["max_seq_length"],
        "random_seed": seed,
        "statistics": stats,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - train_data.pkl ({len(train_texts)} samples)")
    logger.info(f"  - val_data.pkl ({len(val_texts)} samples)")
    logger.info(f"  - test_data.pkl ({len(test_texts)} samples)")
    logger.info(f"  - vocab.json (vocabulary)")
    logger.info(f"  - metadata.json (metadata)")


if __name__ == "__main__":
    main()
