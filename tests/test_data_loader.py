"""
Tests for data loading module.
"""

import numpy as np
import pytest

from src.data_loader import (
    LABEL_MAP,
    REVERSE_LABEL_MAP,
    get_data_statistics,
    load_imdb_data,
    split_data,
)


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_returns_texts_and_labels(self, mock_data_dir):
        """Test that load returns both texts and labels."""
        train_dir = mock_data_dir / "train"
        texts, labels = load_imdb_data(str(train_dir))

        assert isinstance(texts, list)
        assert isinstance(labels, list)
        assert len(texts) == len(labels)

    def test_load_correct_count(self, mock_data_dir):
        """Test that correct number of samples are loaded."""
        train_dir = mock_data_dir / "train"
        texts, labels = load_imdb_data(str(train_dir))

        # 5 positive + 5 negative from fixture
        assert len(texts) == 10

    def test_load_correct_labels(self, mock_data_dir):
        """Test that labels are correctly assigned."""
        train_dir = mock_data_dir / "train"
        texts, labels = load_imdb_data(str(train_dir))

        assert LABEL_MAP["pos"] in labels
        assert LABEL_MAP["neg"] in labels


class TestLabelMaps:
    """Tests for label mapping constants."""

    def test_label_map_values(self):
        """Test label map has correct values."""
        assert LABEL_MAP["neg"] == 0
        assert LABEL_MAP["pos"] == 1

    def test_reverse_label_map_values(self):
        """Test reverse label map has correct values."""
        assert REVERSE_LABEL_MAP[0] == "negative"
        assert REVERSE_LABEL_MAP[1] == "positive"


class TestDataStatistics:
    """Tests for data statistics computation."""

    def test_statistics_returns_dict(self, sample_texts, sample_labels):
        """Test that statistics returns dictionary."""
        stats = get_data_statistics(sample_texts, sample_labels)
        assert isinstance(stats, dict)

    def test_statistics_has_required_keys(self, sample_texts, sample_labels):
        """Test that statistics has required keys."""
        stats = get_data_statistics(sample_texts, sample_labels)

        assert "total" in stats
        assert "positive" in stats
        assert "negative" in stats
        assert "avg_words" in stats

    def test_statistics_correct_total(self, sample_texts, sample_labels):
        """Test that total sample count is correct."""
        stats = get_data_statistics(sample_texts, sample_labels)
        assert stats["total"] == len(sample_texts)


class TestDataSplitting:
    """Tests for data splitting functionality."""

    def test_split_returns_four_lists(self, sample_texts, sample_labels):
        """Test that split returns four lists."""
        result = split_data(sample_texts, sample_labels, validation_split=0.2)

        assert len(result) == 4
        train_texts, train_labels, val_texts, val_labels = result

        assert isinstance(train_texts, list)
        assert isinstance(train_labels, list)
        assert isinstance(val_texts, list)
        assert isinstance(val_labels, list)

    def test_split_preserves_total_count(self, sample_texts, sample_labels):
        """Test that split preserves total sample count."""
        train_texts, train_labels, val_texts, val_labels = split_data(
            sample_texts, sample_labels, validation_split=0.2
        )

        assert len(train_texts) + len(val_texts) == len(sample_texts)

    def test_split_reproducible(self, sample_texts, sample_labels):
        """Test that split is reproducible with same seed."""
        result1 = split_data(sample_texts, sample_labels, random_seed=42)
        result2 = split_data(sample_texts, sample_labels, random_seed=42)

        assert result1[0] == result2[0]
        assert result1[1] == result2[1]
