"""
Text preprocessing module for sentiment classification.
"""

import html
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessor for sentiment classification."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_vocab_size: int = 20000, max_seq_length: int = 256, min_word_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.min_word_freq = min_word_freq
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        self.is_fitted = False

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = html.unescape(str(text))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = text.lower()

        for old, new in [("n't", " not"), ("'re", " are"), ("'s", " is"),
                         ("'d", " would"), ("'ll", " will"), ("'ve", " have")]:
            text = text.replace(old, new)

        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return self.clean_text(text).split()

    def fit(self, texts: list[str]) -> "TextPreprocessor":
        """Build vocabulary from texts."""
        logger.info(f"Building vocabulary from {len(texts)} texts")
        word_freq = Counter()
        for text in texts:
            word_freq.update(self.tokenize(text))

        # Filter by frequency and limit vocab size
        vocab_words = [w for w, f in word_freq.most_common(self.max_vocab_size - 2) if f >= self.min_word_freq]

        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.is_fitted = True
        logger.info(f"Vocabulary size: {len(self.word2idx)}")
        return self

    def transform(self, text: str) -> list[int]:
        """Convert text to sequence of indices."""
        return [self.word2idx.get(t, self.UNK_IDX) for t in self.tokenize(text)]

    def transform_batch(self, texts: list[str]) -> np.ndarray:
        """Convert batch of texts to padded sequences."""
        return self.pad_sequences([self.transform(t) for t in texts])

    def pad_sequences(self, sequences: list[list[int]]) -> np.ndarray:
        """Pad sequences to max_seq_length."""
        padded = np.full((len(sequences), self.max_seq_length), self.PAD_IDX, dtype=np.int64)
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_seq_length)
            padded[i, :length] = seq[:length]
        return padded

    def get_vocab_size(self) -> int:
        return len(self.word2idx)

    def save(self, path: str) -> None:
        """Save preprocessor state."""
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
            json.dump(state, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TextPreprocessor":
        """Load preprocessor state."""
        with open(Path(path) / "preprocessor.json", encoding="utf-8") as f:
            state = json.load(f)
        preprocessor = cls(state["max_vocab_size"], state["max_seq_length"], state["min_word_freq"])
        preprocessor.word2idx = state["word2idx"]
        preprocessor.idx2word = {int(v): k for k, v in state["word2idx"].items()}
        preprocessor.is_fitted = state["is_fitted"]
        return preprocessor
