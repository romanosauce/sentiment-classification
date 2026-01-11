"""
Tests for inference module.
"""

import pytest
import torch

from src.inference import (
    SENTIMENT_LABELS,
    PredictionResult,
    SentimentPredictor,
)


class TestSentimentLabels:
    """Tests for sentiment label constants."""

    def test_labels_have_both_classes(self):
        """Test that both sentiment classes are defined."""
        assert 0 in SENTIMENT_LABELS
        assert 1 in SENTIMENT_LABELS

    def test_label_names(self):
        """Test that label names are correct."""
        assert SENTIMENT_LABELS[0] == "negative"
        assert SENTIMENT_LABELS[1] == "positive"


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""

    def test_creation(self):
        """Test creating PredictionResult."""
        result = PredictionResult(
            text="Test text",
            sentiment="positive",
            confidence=0.95,
            confidence_level="high",
            probabilities={"positive": 0.95, "negative": 0.05},
            inference_time_ms=10.5,
        )

        assert result.sentiment == "positive"
        assert result.confidence == 0.95
        assert result.confidence_level == "high"

    def test_probabilities(self):
        """Test probabilities are stored correctly."""
        result = PredictionResult(
            text="Test",
            sentiment="positive",
            confidence=0.9,
            confidence_level="high",
            probabilities={"positive": 0.9, "negative": 0.1},
            inference_time_ms=5.0,
        )

        assert result.probabilities["positive"] == 0.9
        assert result.probabilities["negative"] == 0.1


class TestSentimentPredictor:
    """Tests for SentimentPredictor class."""

    def test_predict_returns_result(self, model, preprocessor, device):
        """Test that predict returns PredictionResult."""
        predictor = SentimentPredictor(model, preprocessor, device)
        result = predictor.predict("This is a great movie!")

        assert isinstance(result, PredictionResult)

    def test_predict_has_valid_sentiment(self, model, preprocessor, device):
        """Test that prediction has valid sentiment label."""
        predictor = SentimentPredictor(model, preprocessor, device)
        result = predictor.predict("Test movie review")

        assert result.sentiment in ["positive", "negative"]

    def test_predict_has_valid_confidence(self, model, preprocessor, device):
        """Test that prediction has valid confidence."""
        predictor = SentimentPredictor(model, preprocessor, device)
        result = predictor.predict("Test movie review")

        assert 0 <= result.confidence <= 1

    def test_predict_has_confidence_level(self, model, preprocessor, device):
        """Test that prediction has confidence level."""
        predictor = SentimentPredictor(model, preprocessor, device)
        result = predictor.predict("Test movie review")

        assert result.confidence_level in ["high", "medium", "low"]

    def test_predict_has_inference_time(self, model, preprocessor, device):
        """Test that prediction includes inference time."""
        predictor = SentimentPredictor(model, preprocessor, device)
        result = predictor.predict("Test movie review")

        assert result.inference_time_ms > 0

    def test_predict_batch_returns_list(self, model, preprocessor, device, sample_texts):
        """Test that predict_batch returns list of results."""
        predictor = SentimentPredictor(model, preprocessor, device)
        results = predictor.predict_batch(sample_texts)

        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_from_pretrained(self, model, preprocessor, temp_dir, device):
        """Test loading predictor from saved model."""
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        preprocessor.save(str(save_path))

        predictor = SentimentPredictor.from_pretrained(str(save_path), device)
        result = predictor.predict("Test text")

        assert isinstance(result, PredictionResult)


class TestPredictionConsistency:
    """Tests for prediction consistency."""

    def test_same_input_same_output(self, model, preprocessor, device):
        """Test that same input produces same output."""
        predictor = SentimentPredictor(model, preprocessor, device)

        text = "This is a consistent test"
        result1 = predictor.predict(text)
        result2 = predictor.predict(text)

        assert result1.sentiment == result2.sentiment
        assert result1.confidence == result2.confidence

    def test_batch_single_consistency(self, model, preprocessor, device):
        """Test that batch and single predictions are consistent."""
        predictor = SentimentPredictor(model, preprocessor, device)

        text = "Test movie review"
        single_result = predictor.predict(text)
        batch_result = predictor.predict_batch([text])[0]

        assert single_result.sentiment == batch_result.sentiment
        assert abs(single_result.confidence - batch_result.confidence) < 0.01
