"""
Tests for neural network model module.

Tests cover:
- Model initialization
- Forward pass
- Output shapes
- Save/load functionality
- Prediction functionality
"""

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model import SentimentClassifier, create_model_from_config


class TestModelInitialization:
    """Tests for model initialization."""
    
    def test_model_creates_successfully(self):
        """Test that model can be created."""
        model = SentimentClassifier(vocab_size=1000)
        assert model is not None
    
    def test_model_has_embedding(self):
        """Test that model has embedding layer."""
        model = SentimentClassifier(vocab_size=1000)
        assert hasattr(model, "embedding")
        assert isinstance(model.embedding, nn.Embedding)
    
    def test_model_has_lstm(self):
        """Test that model has LSTM layer."""
        model = SentimentClassifier(vocab_size=1000)
        assert hasattr(model, "lstm")
        assert isinstance(model.lstm, nn.LSTM)
    
    def test_model_embedding_size(self):
        """Test embedding layer has correct vocabulary size."""
        vocab_size = 5000
        model = SentimentClassifier(vocab_size=vocab_size)
        assert model.embedding.num_embeddings == vocab_size
    
    def test_model_embedding_dim(self):
        """Test embedding dimension is set correctly."""
        embedding_dim = 64
        model = SentimentClassifier(vocab_size=1000, embedding_dim=embedding_dim)
        assert model.embedding.embedding_dim == embedding_dim
    
    def test_model_bidirectional_setting(self):
        """Test bidirectional LSTM setting."""
        model_bi = SentimentClassifier(vocab_size=1000, bidirectional=True)
        model_uni = SentimentClassifier(vocab_size=1000, bidirectional=False)
        
        assert model_bi.lstm.bidirectional
        assert not model_uni.lstm.bidirectional
    
    def test_model_hidden_dim(self):
        """Test hidden dimension setting."""
        hidden_dim = 128
        model = SentimentClassifier(vocab_size=1000, hidden_dim=hidden_dim)
        assert model.lstm.hidden_size == hidden_dim
    
    def test_model_num_layers(self):
        """Test number of LSTM layers."""
        num_layers = 3
        model = SentimentClassifier(vocab_size=1000, num_layers=num_layers)
        assert model.lstm.num_layers == num_layers
    
    def test_model_stores_config(self):
        """Test that model stores configuration."""
        model = SentimentClassifier(
            vocab_size=1000,
            embedding_dim=64,
            hidden_dim=128,
        )
        assert model.vocab_size == 1000
        assert model.embedding_dim == 64
        assert model.hidden_dim == 128


class TestForwardPass:
    """Tests for model forward pass."""
    
    def test_forward_returns_tensor(self, model, sample_batch):
        """Test that forward pass returns tensor."""
        output = model(sample_batch["input_ids"])
        assert isinstance(output, torch.Tensor)
    
    def test_forward_output_shape(self, model, sample_batch):
        """Test output tensor shape."""
        batch_size = sample_batch["input_ids"].shape[0]
        output = model(sample_batch["input_ids"])
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == 2  # num_classes
    
    def test_forward_with_attention_mask(self, model, sample_batch):
        """Test forward pass with attention mask."""
        output = model(
            sample_batch["input_ids"],
            sample_batch["attention_mask"],
        )
        assert output.shape[1] == 2
    
    def test_forward_deterministic_in_eval(self, model, sample_batch):
        """Test that forward pass is deterministic in eval mode."""
        model.eval()
        
        output1 = model(sample_batch["input_ids"])
        output2 = model(sample_batch["input_ids"])
        
        assert torch.allclose(output1, output2)
    
    def test_forward_different_batch_sizes(self, model, preprocessor):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 16]:
            input_ids = torch.randint(
                0, preprocessor.get_vocab_size(),
                (batch_size, 32)
            )
            output = model(input_ids)
            assert output.shape[0] == batch_size


class TestOutputs:
    """Tests for model outputs."""
    
    def test_output_is_logits(self, model, sample_batch):
        """Test that output can be used as logits."""
        output = model(sample_batch["input_ids"])
        
        # Should be able to apply softmax
        probs = torch.softmax(output, dim=-1)
        
        # Probabilities should sum to 1
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_output_dtype(self, model, sample_batch):
        """Test output tensor dtype."""
        output = model(sample_batch["input_ids"])
        assert output.dtype == torch.float32
    
    def test_output_requires_grad_in_train(self, model, sample_batch):
        """Test that output requires grad in training mode."""
        model.train()
        output = model(sample_batch["input_ids"])
        assert output.requires_grad


class TestPrediction:
    """Tests for prediction functionality."""
    
    def test_predict_returns_dict(self, model, sample_batch):
        """Test that predict returns dictionary."""
        result = model.predict(sample_batch["input_ids"])
        assert isinstance(result, dict)
    
    def test_predict_has_required_keys(self, model, sample_batch):
        """Test that predict result has required keys."""
        result = model.predict(sample_batch["input_ids"])
        
        assert "logits" in result
        assert "probabilities" in result
        assert "predictions" in result
    
    def test_predict_probabilities_sum_to_one(self, model, sample_batch):
        """Test that predicted probabilities sum to 1."""
        result = model.predict(sample_batch["input_ids"])
        probs = result["probabilities"]
        
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_predict_predictions_are_valid_classes(self, model, sample_batch):
        """Test that predictions are valid class indices."""
        result = model.predict(sample_batch["input_ids"])
        preds = result["predictions"]
        
        assert (preds >= 0).all()
        assert (preds < 2).all()  # num_classes = 2
    
    def test_predict_sets_eval_mode(self, model, sample_batch):
        """Test that predict sets model to eval mode."""
        model.train()
        model.predict(sample_batch["input_ids"])
        assert not model.training


class TestSaveLoad:
    """Tests for model save/load functionality."""
    
    def test_save_creates_files(self, model, temp_dir):
        """Test that save creates necessary files."""
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        
        assert (save_path / "pytorch_model.bin").exists()
        assert (save_path / "config.json").exists()
    
    def test_save_config_is_valid_json(self, model, temp_dir):
        """Test that saved config is valid JSON."""
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        
        with open(save_path / "config.json") as f:
            config = json.load(f)
        
        assert "vocab_size" in config
        assert "embedding_dim" in config
    
    def test_load_creates_model(self, model, temp_dir):
        """Test that load creates model successfully."""
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        
        loaded = SentimentClassifier.from_pretrained(str(save_path))
        assert loaded is not None
    
    def test_load_preserves_config(self, model, temp_dir):
        """Test that loaded model has same configuration."""
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        
        loaded = SentimentClassifier.from_pretrained(str(save_path))
        
        assert loaded.vocab_size == model.vocab_size
        assert loaded.embedding_dim == model.embedding_dim
        assert loaded.hidden_dim == model.hidden_dim
        assert loaded.bidirectional == model.bidirectional
    
    def test_load_produces_same_output(self, model, sample_batch, temp_dir):
        """Test that loaded model produces same output."""
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        
        loaded = SentimentClassifier.from_pretrained(str(save_path))
        
        model.eval()
        loaded.eval()
        
        with torch.no_grad():
            original_output = model(sample_batch["input_ids"])
            loaded_output = loaded(sample_batch["input_ids"])
        
        assert torch.allclose(original_output, loaded_output)


class TestModelFromConfig:
    """Tests for creating model from configuration."""
    
    def test_create_from_config(self):
        """Test creating model from config dict."""
        config = {
            "model": {
                "embedding_dim": 64,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": True,
            }
        }
        
        model = create_model_from_config(config, vocab_size=1000)
        
        assert model.embedding_dim == 64
        assert model.hidden_dim == 128
        assert model.num_layers == 2
    
    def test_create_with_defaults(self):
        """Test creating model with default config."""
        config = {"model": {}}
        model = create_model_from_config(config, vocab_size=1000)
        
        assert model is not None
        assert model.vocab_size == 1000


class TestGradientFlow:
    """Tests for gradient flow through model."""
    
    def test_gradients_flow_to_embedding(self, model, sample_batch):
        """Test that gradients flow to embedding layer."""
        model.train()
        output = model(sample_batch["input_ids"])
        loss = output.sum()
        loss.backward()
        
        assert model.embedding.weight.grad is not None
    
    def test_gradients_flow_to_lstm(self, model, sample_batch):
        """Test that gradients flow to LSTM layer."""
        model.train()
        output = model(sample_batch["input_ids"])
        loss = output.sum()
        loss.backward()
        
        assert model.lstm.weight_ih_l0.grad is not None
    
    def test_no_nan_gradients(self, model, sample_batch):
        """Test that gradients don't contain NaN."""
        model.train()
        output = model(sample_batch["input_ids"])
        loss = output.sum()
        loss.backward()
        
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
