"""
Tests for text preprocessing module.

Tests cover:
- Text cleaning
- Tokenization
- Vocabulary building
- Sequence conversion
- Edge cases and validation
"""

import numpy as np
import pytest

from src.preprocessing import (
    TextPreprocessor,
    get_text_statistics,
    validate_text,
)


class TestTextCleaning:
    """Tests for text cleaning functionality."""
    
    def test_clean_html_tags(self):
        """Test removal of HTML tags."""
        preprocessor = TextPreprocessor()
        text = "<p>Hello <b>world</b></p>"
        cleaned = preprocessor.clean_text(text)
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "hello" in cleaned
        assert "world" in cleaned
    
    def test_clean_urls(self):
        """Test removal of URLs."""
        preprocessor = TextPreprocessor()
        text = "Check out https://example.com for more info"
        cleaned = preprocessor.clean_text(text)
        assert "https" not in cleaned
        assert "example" not in cleaned
    
    def test_clean_email(self):
        """Test removal of email addresses."""
        preprocessor = TextPreprocessor()
        text = "Contact me at test@example.com"
        cleaned = preprocessor.clean_text(text)
        assert "@" not in cleaned
    
    def test_lowercase_conversion(self):
        """Test conversion to lowercase."""
        preprocessor = TextPreprocessor()
        text = "HELLO World"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "hello world"
    
    def test_contractions_expansion(self):
        """Test expansion of contractions."""
        preprocessor = TextPreprocessor()
        text = "I don't think it's good"
        cleaned = preprocessor.clean_text(text)
        assert "not" in cleaned
        assert "is" in cleaned
    
    def test_special_characters_removal(self):
        """Test removal of special characters."""
        preprocessor = TextPreprocessor()
        text = "Hello! How are you? #great @mention"
        cleaned = preprocessor.clean_text(text)
        assert "!" not in cleaned
        assert "?" not in cleaned
        assert "#" not in cleaned
        assert "@" not in cleaned
    
    def test_empty_string(self):
        """Test handling of empty string."""
        preprocessor = TextPreprocessor()
        cleaned = preprocessor.clean_text("")
        assert cleaned == ""
    
    def test_none_input(self):
        """Test handling of None input."""
        preprocessor = TextPreprocessor()
        cleaned = preprocessor.clean_text(None)
        assert cleaned == ""
    
    def test_whitespace_normalization(self):
        """Test normalization of multiple whitespaces."""
        preprocessor = TextPreprocessor()
        text = "Hello    world   test"
        cleaned = preprocessor.clean_text(text)
        assert "  " not in cleaned


class TestTokenization:
    """Tests for tokenization functionality."""
    
    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize("This is a test sentence")
        assert tokens == ["this", "is", "a", "test", "sentence"]
    
    def test_tokenization_preserves_words(self):
        """Test that tokenization preserves word content."""
        preprocessor = TextPreprocessor()
        text = "Movie was great"
        tokens = preprocessor.tokenize(text)
        assert "movie" in tokens
        assert "great" in tokens


class TestVocabularyBuilding:
    """Tests for vocabulary building."""
    
    def test_fit_builds_vocabulary(self, sample_texts):
        """Test that fit() builds vocabulary correctly."""
        preprocessor = TextPreprocessor(max_vocab_size=100, min_word_freq=1)
        preprocessor.fit(sample_texts)
        
        assert preprocessor.is_fitted
        assert len(preprocessor.word2idx) > 2  # More than just PAD and UNK
    
    def test_special_tokens_present(self, sample_texts):
        """Test that special tokens are in vocabulary."""
        preprocessor = TextPreprocessor()
        preprocessor.fit(sample_texts)
        
        assert TextPreprocessor.PAD_TOKEN in preprocessor.word2idx
        assert TextPreprocessor.UNK_TOKEN in preprocessor.word2idx
        assert preprocessor.word2idx[TextPreprocessor.PAD_TOKEN] == 0
        assert preprocessor.word2idx[TextPreprocessor.UNK_TOKEN] == 1
    
    def test_vocab_size_limit(self, sample_texts):
        """Test that vocabulary size is limited correctly."""
        max_size = 10
        preprocessor = TextPreprocessor(max_vocab_size=max_size, min_word_freq=1)
        preprocessor.fit(sample_texts)
        
        assert len(preprocessor.word2idx) <= max_size
    
    def test_min_freq_filtering(self):
        """Test minimum frequency filtering."""
        texts = ["hello world", "hello test", "world test"]
        preprocessor = TextPreprocessor(max_vocab_size=100, min_word_freq=2)
        preprocessor.fit(texts)
        
        # Words appearing at least twice should be in vocab
        assert "hello" in preprocessor.word2idx or "world" in preprocessor.word2idx


class TestSequenceConversion:
    """Tests for text-to-sequence conversion."""
    
    def test_transform_returns_indices(self, preprocessor):
        """Test that transform returns list of indices."""
        indices = preprocessor.transform("test movie")
        
        assert isinstance(indices, list)
        assert all(isinstance(i, int) for i in indices)
    
    def test_unknown_words_get_unk_idx(self):
        """Test that unknown words get UNK index."""
        preprocessor = TextPreprocessor(min_word_freq=1)
        preprocessor.fit(["hello world"])
        
        indices = preprocessor.transform("unknown word xyz")
        assert TextPreprocessor.UNK_IDX in indices
    
    def test_transform_without_fit_raises(self):
        """Test that transform without fit raises error."""
        preprocessor = TextPreprocessor()
        
        with pytest.raises(RuntimeError):
            preprocessor.transform("test text")
    
    def test_pad_sequences_correct_shape(self, preprocessor):
        """Test that padding produces correct shape."""
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        padded = preprocessor.pad_sequences(sequences)
        
        assert padded.shape[0] == 3  # num sequences
        assert padded.shape[1] == preprocessor.max_seq_length
    
    def test_pad_sequences_truncates_long(self, preprocessor):
        """Test that long sequences are truncated."""
        long_seq = list(range(1000))
        padded = preprocessor.pad_sequences([long_seq])
        
        assert padded.shape[1] == preprocessor.max_seq_length
    
    def test_transform_batch_returns_numpy(self, preprocessor, sample_texts):
        """Test that transform_batch returns numpy array."""
        result = preprocessor.transform_batch(sample_texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(sample_texts)
        assert result.shape[1] == preprocessor.max_seq_length


class TestSaveLoad:
    """Tests for save/load functionality."""
    
    def test_save_load_roundtrip(self, preprocessor, temp_dir):
        """Test that save and load produce equivalent preprocessor."""
        save_path = temp_dir / "preprocessor"
        
        # Save
        preprocessor.save(str(save_path))
        
        # Load
        loaded = TextPreprocessor.load(str(save_path))
        
        # Verify
        assert loaded.max_vocab_size == preprocessor.max_vocab_size
        assert loaded.max_seq_length == preprocessor.max_seq_length
        assert loaded.word2idx == preprocessor.word2idx
        assert loaded.is_fitted == preprocessor.is_fitted
    
    def test_loaded_preprocessor_transforms(self, preprocessor, temp_dir):
        """Test that loaded preprocessor can transform text."""
        save_path = temp_dir / "preprocessor"
        preprocessor.save(str(save_path))
        loaded = TextPreprocessor.load(str(save_path))
        
        original = preprocessor.transform("test movie great")
        loaded_result = loaded.transform("test movie great")
        
        assert original == loaded_result


class TestValidation:
    """Tests for input validation."""
    
    def test_validate_text_valid(self):
        """Test validation of valid text."""
        is_valid, error = validate_text("Valid text here")
        assert is_valid
        assert error == ""
    
    def test_validate_text_none(self):
        """Test validation of None."""
        is_valid, error = validate_text(None)
        assert not is_valid
        assert "None" in error
    
    def test_validate_text_not_string(self):
        """Test validation of non-string input."""
        is_valid, error = validate_text(123)
        assert not is_valid
        assert "string" in error
    
    def test_validate_text_empty(self):
        """Test validation of empty string."""
        is_valid, error = validate_text("")
        assert not is_valid
        assert "empty" in error
    
    def test_validate_text_too_long(self):
        """Test validation of text exceeding max length."""
        long_text = "a" * 60000
        is_valid, error = validate_text(long_text)
        assert not is_valid
        assert "maximum length" in error


class TestTextStatistics:
    """Tests for text statistics calculation."""
    
    def test_statistics_returns_dict(self, sample_texts):
        """Test that statistics returns dictionary."""
        stats = get_text_statistics(sample_texts)
        
        assert isinstance(stats, dict)
        assert "total_texts" in stats
        assert "avg_word_count" in stats
    
    def test_statistics_correct_count(self, sample_texts):
        """Test that total count is correct."""
        stats = get_text_statistics(sample_texts)
        assert stats["total_texts"] == len(sample_texts)
    
    def test_statistics_empty_list(self):
        """Test statistics with empty list."""
        stats = get_text_statistics([])
        assert "error" in stats
