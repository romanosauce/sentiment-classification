"""
Neural network model for sentiment classification.

Implements LSTM-based architecture with optional bidirectionality
and attention mechanism.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sentiment_classifier")


class SentimentClassifier(nn.Module):
    """
    LSTM-based sentiment classifier.
    
    Architecture:
    - Embedding layer
    - LSTM/BiLSTM layers
    - Optional attention mechanism
    - Fully connected classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2,
        padding_idx: int = 0,
    ):
        """
        Initialize the model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            num_classes: Number of output classes
            padding_idx: Index of padding token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self._init_weights()
        
        logger.info(f"Model initialized with {self._count_parameters():,} parameters")
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch_size, seq_length]
            attention_mask: Mask for padding [batch_size, seq_length]
            
        Returns:
            Logits tensor [batch_size, num_classes]
        """
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        return logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Make predictions with probabilities.
        
        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            
        Returns:
            Dictionary with 'logits', 'probabilities', and 'predictions'
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
        }
    
    def save_pretrained(self, save_path: str) -> None:
        """
        Save model in Hugging Face compatible format.
        
        Args:
            save_path: Directory to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "bidirectional": self.bidirectional,
            "num_classes": self.num_classes,
            "padding_idx": self.padding_idx,
            "model_type": "lstm_sentiment_classifier",
        }
        
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: torch.device | None = None) -> "SentimentClassifier":
        """
        Load model from Hugging Face compatible format.
        
        Args:
            load_path: Directory containing the model
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        load_path = Path(load_path)
        
        with open(load_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        config.pop("model_type", None)
        
        model = cls(**config)
        
        state_dict = torch.load(
            load_path / "pytorch_model.bin",
            map_location=device or torch.device("cpu"),
        )
        model.load_state_dict(state_dict)
        
        if device:
            model = model.to(device)
        
        logger.info(f"Model loaded from {load_path}")
        return model


def create_model_from_config(config: dict[str, Any], vocab_size: int) -> SentimentClassifier:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Model configuration
        vocab_size: Vocabulary size
        
    Returns:
        Initialized model
    """
    model_config = config.get("model", {})
    
    return SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=model_config.get("embedding_dim", 128),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 2),
        dropout=model_config.get("dropout", 0.3),
        bidirectional=model_config.get("bidirectional", True),
        num_classes=2,
    )
