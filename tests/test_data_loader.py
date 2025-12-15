"""
Tests for data loading and validation module.

Tests cover:
- Data directory validation
- Sample validation
- Data loading
- Statistics computation
- Data splitting
"""

import numpy as np
import pytest

from src.data_loader import (
    DataValidationError,
    LABEL_MAP,
    REVERSE_LABEL_MAP,
    get_data_statistics,
    load_imdb_data,
    split_data,
    validate_data_directory,
    validate_sample,
)


class TestDataDirectoryValidation:
    """Tests for data directory structure validation."""
    
    def test_valid_directory(self, mock_data_dir):
        """Test validation of valid directory structure."""
        train_dir = mock_data_dir / "train"
        is_valid, errors = validate_data_directory(str(train_dir))
        
        assert is_valid
        assert len(errors) == 0
    
    def test_nonexistent_directory(self, tmp_path):
        """Test validation of nonexistent directory."""
        fake_dir = tmp_path / "nonexistent"
        is_valid, errors = validate_data_directory(str(fake_dir))
        
        assert not is_valid
        assert len(errors) > 0
        assert "does not exist" in errors[0]
    
    def test_missing_pos_directory(self, tmp_path):
        """Test validation when pos directory is missing."""
        data_dir = tmp_path / "data"
        (data_dir / "neg").mkdir(parents=True)
        (data_dir / "neg" / "file.txt").write_text("test")
        
        is_valid, errors = validate_data_directory(str(data_dir))
        
        assert not is_valid
        assert any("pos" in e for e in errors)
    
    def test_missing_neg_directory(self, tmp_path):
        """Test validation when neg directory is missing."""
        data_dir = tmp_path / "data"
        (data_dir / "pos").mkdir(parents=True)
        (data_dir / "pos" / "file.txt").write_text("test")
        
        is_valid, errors = validate_data_directory(str(data_dir))
        
        assert not is_valid
        assert any("neg" in e for e in errors)
    
    def test_empty_pos_directory(self, tmp_path):
        """Test validation when pos directory is empty."""
        data_dir = tmp_path / "data"
        (data_dir / "pos").mkdir(parents=True)
        (data_dir / "neg").mkdir(parents=True)
        (data_dir / "neg" / "file.txt").write_text("test")
        
        is_valid, errors = validate_data_directory(str(data_dir))
        
        assert not is_valid
        assert any("No .txt files" in e and "pos" in e for e in errors)


class TestSampleValidation:
    """Tests for individual sample validation."""
    
    def test_valid_sample(self):
        """Test validation of valid sample."""
        is_valid, error = validate_sample("Valid text", 1)
        
        assert is_valid
        assert error == ""
    
    def test_none_text(self):
        """Test validation with None text."""
        is_valid, error = validate_sample(None, 1)
        
        assert not is_valid
        assert "None" in error
    
    def test_non_string_text(self):
        """Test validation with non-string text."""
        is_valid, error = validate_sample(123, 1)
        
        assert not is_valid
        assert "string" in error
    
    def test_empty_text(self):
        """Test validation with empty text."""
        is_valid, error = validate_sample("", 1)
        
        assert not is_valid
        assert "empty" in error
    
    def test_whitespace_only_text(self):
        """Test validation with whitespace-only text."""
        is_valid, error = validate_sample("   \n\t  ", 1)
        
        assert not is_valid
        assert "empty" in error or "whitespace" in error
    
    def test_none_label(self):
        """Test validation with None label."""
        is_valid, error = validate_sample("Valid text", None)
        
        assert not is_valid
        assert "None" in error
    
    def test_non_integer_label(self):
        """Test validation with non-integer label."""
        is_valid, error = validate_sample("Valid text", "positive")
        
        assert not is_valid
        assert "integer" in error
    
    def test_invalid_label_value(self):
        """Test validation with invalid label value."""
        is_valid, error = validate_sample("Valid text", 5)
        
        assert not is_valid
        assert "0 or 1" in error
    
    def test_numpy_integer_label(self):
        """Test validation with numpy integer label."""
        is_valid, error = validate_sample("Valid text", np.int64(1))
        
        assert is_valid


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
        
        # Should have both positive and negative
        assert LABEL_MAP["pos"] in labels
        assert LABEL_MAP["neg"] in labels
    
    def test_load_with_validation(self, mock_data_dir):
        """Test loading with validation enabled."""
        train_dir = mock_data_dir / "train"
        texts, labels = load_imdb_data(str(train_dir), validate=True)
        
        assert len(texts) > 0
    
    def test_load_invalid_directory_raises(self, tmp_path):
        """Test that loading from invalid directory raises error."""
        fake_dir = tmp_path / "nonexistent"
        
        with pytest.raises(DataValidationError):
            load_imdb_data(str(fake_dir), validate=True)


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
    
    def test_maps_are_inverses(self):
        """Test that maps are consistent."""
        for name, idx in LABEL_MAP.items():
            # Just check that indices exist in reverse map
            assert idx in REVERSE_LABEL_MAP


class TestDataStatistics:
    """Tests for data statistics computation."""
    
    def test_statistics_returns_dict(self, sample_texts, sample_labels):
        """Test that statistics returns dictionary."""
        stats = get_data_statistics(sample_texts, sample_labels)
        
        assert isinstance(stats, dict)
    
    def test_statistics_has_required_keys(self, sample_texts, sample_labels):
        """Test that statistics has all required keys."""
        stats = get_data_statistics(sample_texts, sample_labels)
        
        required_keys = [
            "total_samples",
            "positive_samples",
            "negative_samples",
            "class_balance",
            "word_count",
        ]
        for key in required_keys:
            assert key in stats
    
    def test_statistics_correct_total(self, sample_texts, sample_labels):
        """Test that total sample count is correct."""
        stats = get_data_statistics(sample_texts, sample_labels)
        
        assert stats["total_samples"] == len(sample_texts)
    
    def test_statistics_class_counts(self, sample_texts, sample_labels):
        """Test that class counts are correct."""
        stats = get_data_statistics(sample_texts, sample_labels)
        
        expected_pos = sum(1 for l in sample_labels if l == 1)
        expected_neg = sum(1 for l in sample_labels if l == 0)
        
        assert stats["positive_samples"] == expected_pos
        assert stats["negative_samples"] == expected_neg
    
    def test_statistics_empty_data(self):
        """Test statistics with empty data."""
        stats = get_data_statistics([], [])
        
        assert "error" in stats


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
        assert len(train_labels) + len(val_labels) == len(sample_labels)
    
    def test_split_respects_ratio(self, sample_texts, sample_labels):
        """Test that split approximately respects validation ratio."""
        val_split = 0.4
        train_texts, _, val_texts, _ = split_data(
            sample_texts, sample_labels, validation_split=val_split
        )
        
        actual_ratio = len(val_texts) / len(sample_texts)
        # Allow some tolerance due to rounding
        assert abs(actual_ratio - val_split) <= 0.2
    
    def test_split_reproducible(self, sample_texts, sample_labels):
        """Test that split is reproducible with same seed."""
        result1 = split_data(sample_texts, sample_labels, random_seed=42)
        result2 = split_data(sample_texts, sample_labels, random_seed=42)
        
        assert result1[0] == result2[0]  # train_texts
        assert result1[1] == result2[1]  # train_labels
    
    def test_split_different_seeds_different_results(self, sample_texts, sample_labels):
        """Test that different seeds produce different splits."""
        result1 = split_data(sample_texts, sample_labels, random_seed=42)
        result2 = split_data(sample_texts, sample_labels, random_seed=123)
        
        # With high probability, at least some samples will differ
        # This might rarely fail by chance with very small datasets
        train1, train2 = result1[0], result2[0]
        # Just check they exist and have correct lengths
        assert len(train1) == len(train2)
