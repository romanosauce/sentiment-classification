"""
Tests for text preprocessing module.
"""

import numpy as np
import pytest

from src.preprocessing import TextPreprocessor


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

    def test_clean_urls(self):
        """Test removal of URLs."""
        preprocessor = TextPreprocessor()
        text = "Check out https://example.com for more info"
        cleaned = preprocessor.clean_text(text)
        assert "https" not in cleaned

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
        text = "Hello! How are you?"
        cleaned = preprocessor.clean_text(text)
        assert "!" not in cleaned
        assert "?" not in cleaned

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
        tokens = preprocessor.tokenize("Movie was great")
        assert "movie" in tokens
        assert "great" in tokens


class TestVocabularyBuilding:
    """Tests for vocabulary building."""

    def test_fit_builds_vocabulary(self, sample_texts):
        """Test that fit() builds vocabulary correctly."""
        preprocessor = TextPreprocessor(max_vocab_size=100, min_word_freq=1)
        preprocessor.fit(sample_texts)

        assert preprocessor.is_fitted
        assert len(preprocessor.word2idx) > 2

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

    def test_pad_sequences_correct_shape(self, preprocessor):
        """Test that padding produces correct shape."""
        sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
        padded = preprocessor.pad_sequences(sequences)

        assert padded.shape[0] == 3
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
        preprocessor.save(str(save_path))

        loaded = TextPreprocessor.load(str(save_path))

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
