"""
Text preprocessing module for sentiment classification.

Handles text cleaning, tokenization, and vocabulary building.
"""

import html
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("sentiment_classifier")


class TextPreprocessor:
    """
    Text preprocessor for sentiment classification.
    
    Handles text cleaning, tokenization, vocabulary building,
    and text-to-sequence conversion.
    """
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1
    
    def __init__(
        self,
        max_vocab_size: int = 20000,
        max_seq_length: int = 256,
        min_word_freq: int = 2,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_seq_length: Maximum sequence length
            min_word_freq: Minimum word frequency to include in vocabulary
        """
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.min_word_freq = min_word_freq
        
        self.word2idx: dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word: dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }
        self.word_freq: Counter = Counter()
        self.is_fitted = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        text = html.unescape(text)
        
        text = re.sub(r"<[^>]+>", " ", text)
        
        text = re.sub(r"http\S+|www\S+", " ", text)
        
        text = re.sub(r"\S+@\S+", " ", text)
        
        text = text.lower()
        
        contractions = {
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of tokens
        """
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        return tokens
    
    def fit(self, texts: list[str]) -> "TextPreprocessor":
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Self for chaining
        """
        logger.info(f"Building vocabulary from {len(texts)} texts...")
        
        self.word_freq = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        logger.info(f"Total unique words: {len(self.word_freq)}")
        
        filtered_words = [
            word for word, freq in self.word_freq.items()
            if freq >= self.min_word_freq
        ]
        logger.info(f"Words after frequency filter: {len(filtered_words)}")
        
        most_common = sorted(filtered_words, key=lambda w: -self.word_freq[w])
        vocab_words = most_common[:self.max_vocab_size - 2]
        
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        self.idx2word = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }
        
        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.is_fitted = True
        logger.info(f"Final vocabulary size: {len(self.word2idx)}")
        
        return self
    
    def transform(self, text: str) -> list[int]:
        """
        Convert text to sequence of indices.
        
        Args:
            text: Text string
            
        Returns:
            List of token indices
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        tokens = self.tokenize(text)
        indices = [
            self.word2idx.get(token, self.UNK_IDX)
            for token in tokens
        ]
        return indices
    
    def transform_batch(self, texts: list[str]) -> np.ndarray:
        """
        Convert batch of texts to padded sequences.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of shape (batch_size, max_seq_length)
        """
        sequences = []
        for text in texts:
            seq = self.transform(text)
            sequences.append(seq)
        
        return self.pad_sequences(sequences)
    
    def pad_sequences(self, sequences: list[list[int]]) -> np.ndarray:
        """
        Pad sequences to same length.
        
        Args:
            sequences: List of token index sequences
            
        Returns:
            Padded numpy array
        """
        padded = np.full(
            (len(sequences), self.max_seq_length),
            self.PAD_IDX,
            dtype=np.int64,
        )
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_seq_length)
            padded[i, :length] = seq[:length]
        
        return padded
    
    def get_vocab_size(self) -> int:
        return len(self.word2idx)
    
    def save(self, path: str) -> None:
        """
        Save preprocessor state.
        
        Args:
            path: Save directory path
        """
        import json
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        state = {
            "max_vocab_size": self.max_vocab_size,
            "max_seq_length": self.max_seq_length,
            "min_word_freq": self.min_word_freq,
            "word2idx": self.word2idx,
            "is_fitted": self.is_fitted,
        }
        
        with open(save_path / "preprocessor.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Preprocessor saved to {save_path}")
    
    @classmethod
    def load(cls, path: str) -> "TextPreprocessor":
        """
        Load preprocessor state.
        
        Args:
            path: Load directory path
            
        Returns:
            Loaded TextPreprocessor instance
        """
        import json
        
        load_path = Path(path)
        
        with open(load_path / "preprocessor.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        
        preprocessor = cls(
            max_vocab_size=state["max_vocab_size"],
            max_seq_length=state["max_seq_length"],
            min_word_freq=state["min_word_freq"],
        )
        preprocessor.word2idx = state["word2idx"]
        preprocessor.idx2word = {int(v): k for k, v in state["word2idx"].items()}
        preprocessor.is_fitted = state["is_fitted"]
        
        logger.info(f"Preprocessor loaded from {load_path}")
        return preprocessor


def validate_text(text: Any) -> tuple[bool, str]:
    """
    Validate input text.
    
    Args:
        text: Input to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if text is None:
        return False, "Text cannot be None"
    
    if not isinstance(text, str):
        return False, f"Text must be string, got {type(text).__name__}"
    
    if len(text) == 0:
        return False, "Text cannot be empty"
    
    if len(text) > 50000:
        return False, "Text exceeds maximum length of 50000 characters"
    
    return True, ""


def get_text_statistics(texts: list[str]) -> dict[str, Any]:
    """
    Compute statistics for a collection of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with text statistics
    """
    lengths = [len(text.split()) for text in texts if isinstance(text, str)]
    
    if not lengths:
        return {"error": "No valid texts found"}
    
    return {
        "total_texts": len(texts),
        "valid_texts": len(lengths),
        "avg_word_count": np.mean(lengths),
        "std_word_count": np.std(lengths),
        "min_word_count": np.min(lengths),
        "max_word_count": np.max(lengths),
        "median_word_count": np.median(lengths),
    }
