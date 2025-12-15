"""
Pytest configuration and fixtures for sentiment classifier tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SentimentClassifier
from src.preprocessing import TextPreprocessor


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing."""
    return [
        "This movie was absolutely fantastic! I loved every minute.",
        "Terrible film. Complete waste of time.",
        "An okay movie, nothing special.",
        "Best movie ever! Highly recommended!",
        "Boring and predictable story.",
    ]


@pytest.fixture
def sample_labels() -> list[int]:
    """Sample labels corresponding to sample_texts."""
    return [1, 0, 0, 1, 0]  # 1 = positive, 0 = negative


@pytest.fixture
def preprocessor(sample_texts) -> TextPreprocessor:
    """Fitted preprocessor for testing."""
    preprocessor = TextPreprocessor(
        max_vocab_size=1000,
        max_seq_length=64,
        min_word_freq=1,
    )
    preprocessor.fit(sample_texts)
    return preprocessor


@pytest.fixture
def model(preprocessor) -> SentimentClassifier:
    """Small model for testing."""
    return SentimentClassifier(
        vocab_size=preprocessor.get_vocab_size(),
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
        num_classes=2,
    )


@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def sample_batch(preprocessor, sample_texts, sample_labels) -> dict:
    """Create a sample batch for testing."""
    sequences = [preprocessor.transform(text) for text in sample_texts]
    padded = preprocessor.pad_sequences(sequences)
    
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = (input_ids != 0).long()
    labels = torch.tensor(sample_labels, dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
    }


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def mock_data_dir(tmp_path) -> Path:
    """Create mock data directory structure."""
    data_dir = tmp_path / "data"
    
    # Create train directory
    train_pos = data_dir / "train" / "pos"
    train_neg = data_dir / "train" / "neg"
    train_pos.mkdir(parents=True)
    train_neg.mkdir(parents=True)
    
    # Create test directory
    test_pos = data_dir / "test" / "pos"
    test_neg = data_dir / "test" / "neg"
    test_pos.mkdir(parents=True)
    test_neg.mkdir(parents=True)
    
    # Create sample files
    for i in range(5):
        (train_pos / f"pos_{i}.txt").write_text(f"Great movie number {i}! Loved it!")
        (train_neg / f"neg_{i}.txt").write_text(f"Terrible movie number {i}. Hated it.")
        (test_pos / f"pos_{i}.txt").write_text(f"Wonderful film number {i}!")
        (test_neg / f"neg_{i}.txt").write_text(f"Awful film number {i}.")
    
    return data_dir
