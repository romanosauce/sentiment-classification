"""
Data loading and validation module for IMDB dataset.

Handles loading data from files, validation, and basic statistics.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("sentiment_classifier")


LABEL_MAP = {"neg": 0, "pos": 1}
REVERSE_LABEL_MAP = {0: "negative", 1: "positive"}


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


def validate_data_directory(data_path: str) -> tuple[bool, list[str]]:
    """
    Validate that data directory has correct structure.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    data_path = Path(data_path)
    
    if not data_path.exists():
        errors.append(f"Data directory does not exist: {data_path}")
        return False, errors
    
    if not data_path.is_dir():
        errors.append(f"Path is not a directory: {data_path}")
        return False, errors
    
    pos_dir = data_path / "pos"
    neg_dir = data_path / "neg"
    
    if not pos_dir.exists():
        errors.append(f"Missing 'pos' directory: {pos_dir}")
    elif not any(pos_dir.glob("*.txt")):
        errors.append(f"No .txt files found in 'pos' directory: {pos_dir}")
    
    if not neg_dir.exists():
        errors.append(f"Missing 'neg' directory: {neg_dir}")
    elif not any(neg_dir.glob("*.txt")):
        errors.append(f"No .txt files found in 'neg' directory: {neg_dir}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_sample(text: Any, label: Any) -> tuple[bool, str]:
    """
    Validate a single data sample.
    
    Args:
        text: Text content
        label: Label value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if text is None:
        return False, "Text is None"
    
    if not isinstance(text, str):
        return False, f"Text must be string, got {type(text).__name__}"
    
    if len(text.strip()) == 0:
        return False, "Text is empty or whitespace only"
    
    if label is None:
        return False, "Label is None"
    
    if not isinstance(label, (int, np.integer)):
        return False, f"Label must be integer, got {type(label).__name__}"
    
    if label not in [0, 1]:
        return False, f"Label must be 0 or 1, got {label}"
    
    return True, ""


def load_imdb_data(
    data_path: str,
    validate: bool = True,
) -> tuple[list[str], list[int]]:
    """
    Load IMDB dataset from directory.
    
    Args:
        data_path: Path to data directory (train or test)
        validate: Whether to validate the data
        
    Returns:
        Tuple of (texts, labels)
        
    Raises:
        DataValidationError: If validation fails
    """
    data_path = Path(data_path)
    
    if validate:
        is_valid, errors = validate_data_directory(data_path)
        if not is_valid:
            raise DataValidationError(
                f"Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
    
    texts = []
    labels = []
    invalid_samples = []
    
    pos_dir = data_path / "pos"
    pos_files = sorted(pos_dir.glob("*.txt"))
    logger.info(f"Loading {len(pos_files)} positive reviews from {pos_dir}")
    
    for file_path in pos_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            label = LABEL_MAP["pos"]
            
            if validate:
                is_valid, error = validate_sample(text, label)
                if not is_valid:
                    invalid_samples.append((file_path.name, error))
                    continue
            
            texts.append(text)
            labels.append(label)
        except Exception as e:
            invalid_samples.append((file_path.name, str(e)))
    
    neg_dir = data_path / "neg"
    neg_files = sorted(neg_dir.glob("*.txt"))
    logger.info(f"Loading {len(neg_files)} negative reviews from {neg_dir}")
    
    for file_path in neg_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            label = LABEL_MAP["neg"]
            
            if validate:
                is_valid, error = validate_sample(text, label)
                if not is_valid:
                    invalid_samples.append((file_path.name, error))
                    continue
            
            texts.append(text)
            labels.append(label)
        except Exception as e:
            invalid_samples.append((file_path.name, str(e)))
    
    if invalid_samples:
        logger.warning(
            f"Found {len(invalid_samples)} invalid samples: "
            + ", ".join(f"{name}: {err}" for name, err in invalid_samples[:5])
        )
    
    logger.info(f"Loaded {len(texts)} valid samples")
    
    return texts, labels


def get_data_statistics(texts: list[str], labels: list[int]) -> dict[str, Any]:
    """
    Compute statistics for the loaded dataset.
    
    Args:
        texts: List of text strings
        labels: List of labels
        
    Returns:
        Dictionary with dataset statistics
    """
    if not texts or not labels:
        return {"error": "Empty dataset"}
    
    labels_array = np.array(labels)
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]
    
    pos_count = int(np.sum(labels_array == 1))
    neg_count = int(np.sum(labels_array == 0))
    
    stats = {
        "total_samples": len(texts),
        "positive_samples": pos_count,
        "negative_samples": neg_count,
        "class_balance": {
            "positive_ratio": pos_count / len(texts),
            "negative_ratio": neg_count / len(texts),
        },
        "word_count": {
            "mean": float(np.mean(word_counts)),
            "std": float(np.std(word_counts)),
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "median": float(np.median(word_counts)),
        },
        "char_count": {
            "mean": float(np.mean(char_counts)),
            "std": float(np.std(char_counts)),
            "min": int(np.min(char_counts)),
            "max": int(np.max(char_counts)),
        },
    }
    
    return stats


def print_data_statistics(stats: dict[str, Any]) -> None:
    """
    Print dataset statistics in a formatted way.
    
    Args:
        stats: Statistics dictionary from get_data_statistics
    """
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    
    print(f"\nTotal samples: {stats['total_samples']}")
    print(f"  - Positive: {stats['positive_samples']} ({stats['class_balance']['positive_ratio']:.1%})")
    print(f"  - Negative: {stats['negative_samples']} ({stats['class_balance']['negative_ratio']:.1%})")
    
    print(f"\nWord count statistics:")
    wc = stats['word_count']
    print(f"  - Mean: {wc['mean']:.1f}")
    print(f"  - Std: {wc['std']:.1f}")
    print(f"  - Min: {wc['min']}")
    print(f"  - Max: {wc['max']}")
    print(f"  - Median: {wc['median']:.1f}")
    
    print("=" * 50 + "\n")


def split_data(
    texts: list[str],
    labels: list[int],
    validation_split: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """
    Split data into training and validation sets.
    
    Args:
        texts: List of text strings
        labels: List of labels
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels)
    """
    np.random.seed(random_seed)
    
    n_samples = len(texts)
    indices = np.random.permutation(n_samples)
    
    n_val = int(n_samples * validation_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    logger.info(f"Split data: {len(train_texts)} train, {len(val_texts)} validation")
    
    return train_texts, train_labels, val_texts, val_labels
