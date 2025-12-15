"""
PyTorch Dataset for sentiment classification.
"""

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import TextPreprocessor

logger = logging.getLogger("sentiment_classifier")


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment classification.
    
    Handles text-to-tensor conversion with preprocessing.
    """
    
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        preprocessor: TextPreprocessor,
        max_seq_length: int | None = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of integer labels (0 or 1)
            preprocessor: Fitted TextPreprocessor instance
            max_seq_length: Override max sequence length (optional)
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of labels ({len(labels)})"
            )
        
        if not preprocessor.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before creating dataset")
        
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_seq_length = max_seq_length or preprocessor.max_seq_length
        
        self._precompute_sequences()
    
    def _precompute_sequences(self) -> None:
        """Pre-compute token sequences for all texts."""
        logger.info(f"Pre-computing sequences for {len(self.texts)} samples...")
        
        self.sequences = []
        for text in self.texts:
            seq = self.preprocessor.transform(text)
            self.sequences.append(seq)
        
        logger.info("Sequences pre-computed")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'label' tensors
        """
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if len(seq) > self.max_seq_length:
            seq = seq[:self.max_seq_length]
        
        attention_mask = [1] * len(seq)
        
        padding_length = self.max_seq_length - len(seq)
        seq = seq + [self.preprocessor.PAD_IDX] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
    
    def get_vocab_size(self) -> int:
        return self.preprocessor.get_vocab_size()
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.
        
        Returns:
            Tensor with class weights
        """
        labels_array = np.array(self.labels)
        class_counts = np.bincount(labels_array, minlength=2)
        total = len(labels_array)
        
        weights = total / (2 * class_counts + 1e-6)
        
        return torch.tensor(weights, dtype=torch.float)


def create_data_loaders(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    preprocessor: TextPreprocessor,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        preprocessor: Fitted preprocessor
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SentimentDataset(train_texts, train_labels, preprocessor)
    val_dataset = SentimentDataset(val_texts, val_labels, preprocessor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(
        f"Created data loaders: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches"
    )
    
    return train_loader, val_loader
