"""
Data loading module for IMDB dataset.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LABEL_MAP = {"neg": 0, "pos": 1}
REVERSE_LABEL_MAP = {0: "negative", 1: "positive"}


def load_imdb_data(data_path: str) -> tuple[list[str], list[int]]:
    """
    Load IMDB dataset from directory.

    Args:
        data_path: Path to data directory (train or test)

    Returns:
        Tuple of (texts, labels)
    """
    data_path = Path(data_path)
    texts, labels = [], []

    for sentiment, label in LABEL_MAP.items():
        files = sorted((data_path / sentiment).glob("*.txt"))
        logger.info(f"Loading {len(files)} {sentiment} reviews")
        for f in files:
            texts.append(f.read_text(encoding="utf-8"))
            labels.append(label)

    logger.info(f"Loaded {len(texts)} samples")
    return texts, labels


def split_data(
    texts: list[str],
    labels: list[int],
    validation_split: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Split data into training and validation sets."""
    np.random.seed(random_seed)
    indices = np.random.permutation(len(texts))
    n_val = int(len(texts) * validation_split)

    val_idx, train_idx = indices[:n_val], indices[n_val:]
    return (
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [texts[i] for i in val_idx],
        [labels[i] for i in val_idx],
    )


def get_data_statistics(texts: list[str], labels: list[int]) -> dict:
    """Compute statistics for the dataset."""
    labels_arr = np.array(labels)
    word_counts = [len(t.split()) for t in texts]

    return {
        "total": len(texts),
        "positive": int((labels_arr == 1).sum()),
        "negative": int((labels_arr == 0).sum()),
        "avg_words": float(np.mean(word_counts)),
    }


def print_data_statistics(stats: dict) -> None:
    """Print dataset statistics."""
    print(f"Samples: {stats['total']} (pos: {stats['positive']}, neg: {stats['negative']})")
    print(f"Avg words: {stats['avg_words']:.1f}")
