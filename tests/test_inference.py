"""
Tests for inference module.

Tests cover:
- Prediction result structure
- Logits to prediction conversion
- Input validation
- SentimentPredictor class
- Confidence levels
"""

import pytest
import torch

from src.inference import (
    CONFIDENCE_THRESHOLDS,
    SENTIMENT_LABELS,
    PredictionResult,
    SentimentPredictor,
    convert_logits_to_prediction,
    get_confidence_level,
    validate_prediction_input,
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


class TestConfidenceLevels:
    """Tests for confidence level functionality."""
    
    def test_high_confidence(self):
        """Test high confidence level."""
        level = get_confidence_level(0.95)
        assert level == "high"
    
    def test_medium_confidence(self):
        """Test medium confidence level."""
        level = get_confidence_level(0.7)
        assert level == "medium"
    
    def test_low_confidence(self):
        """Test low confidence level."""
        level = get_confidence_level(0.5)
        assert level == "low"
    
    def test_boundary_high(self):
        """Test boundary for high confidence."""
        threshold = CONFIDENCE_THRESHOLDS["high"]
        assert get_confidence_level(threshold) == "high"
        assert get_confidence_level(threshold - 0.01) == "medium"
    
    def test_boundary_medium(self):
        """Test boundary for medium confidence."""
        threshold = CONFIDENCE_THRESHOLDS["medium"]
        assert get_confidence_level(threshold) == "medium"
        assert get_confidence_level(threshold - 0.01) == "low"


class TestLogitsConversion:
    """Tests for logits to prediction conversion."""
    
    def test_returns_tuple(self):
        """Test that conversion returns tuple."""
        logits = torch.tensor([[2.0, -1.0]])
        result = convert_logits_to_prediction(logits)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_prediction_is_argmax(self):
        """Test that prediction is argmax of logits."""
        # Class 0 should win
        logits = torch.tensor([[5.0, -5.0]])
        pred, _, _ = convert_logits_to_prediction(logits)
        assert pred == 0
        
        # Class 1 should win
        logits = torch.tensor([[-5.0, 5.0]])
        pred, _, _ = convert_logits_to_prediction(logits)
        assert pred == 1
    
    def test_confidence_is_max_probability(self):
        """Test that confidence is max probability."""
        logits = torch.tensor([[10.0, 0.0]])  # High confidence for class 0
        _, confidence, _ = convert_logits_to_prediction(logits)
        
        assert confidence > 0.99
    
    def test_probabilities_sum_to_one(self):
        """Test that probabilities sum to 1."""
        logits = torch.tensor([[1.0, 2.0]])
        _, _, probs = convert_logits_to_prediction(logits)
        
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-5
    
    def test_probabilities_have_correct_keys(self):
        """Test that probabilities dict has correct keys."""
        logits = torch.tensor([[1.0, 2.0]])
        _, _, probs = convert_logits_to_prediction(logits)
        
        assert "negative" in probs
        assert "positive" in probs
    
    def test_returns_python_types(self):
        """Test that returned values are Python types, not tensors."""
        logits = torch.tensor([[1.0, 2.0]])
        pred, conf, probs = convert_logits_to_prediction(logits)
        
        assert isinstance(pred, int)
        assert isinstance(conf, float)
        assert all(isinstance(v, float) for v in probs.values())


class TestInputValidation:
    """Tests for prediction input validation."""
    
    def test_valid_input(self):
        """Test validation of valid input."""
        is_valid, error = validate_prediction_input("Valid text here")
        assert is_valid
        assert error == ""
    
    def test_none_input(self):
        """Test validation of None input."""
        is_valid, error = validate_prediction_input(None)
        assert not is_valid
        assert "None" in error
    
    def test_non_string_input(self):
        """Test validation of non-string input."""
        is_valid, error = validate_prediction_input(123)
        assert not is_valid
        assert "string" in error
    
    def test_empty_string(self):
        """Test validation of empty string."""
        is_valid, error = validate_prediction_input("")
        assert not is_valid
        assert "empty" in error
    
    def test_whitespace_only(self):
        """Test validation of whitespace-only string."""
        is_valid, error = validate_prediction_input("   \n\t   ")
        assert not is_valid
        assert "empty" in error
    
    def test_very_long_text(self):
        """Test validation of very long text."""
        long_text = "a" * 60000
        is_valid, error = validate_prediction_input(long_text)
        assert not is_valid
        assert "maximum length" in error
    
    def test_max_length_boundary(self):
        """Test boundary of maximum length."""
        # Just under limit should be valid
        text = "a" * 50000
        is_valid, _ = validate_prediction_input(text)
        assert is_valid
        
        # Just over limit should be invalid
        text = "a" * 50001
        is_valid, _ = validate_prediction_input(text)
        assert not is_valid


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
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = PredictionResult(
            text="Test text",
            sentiment="positive",
            confidence=0.95,
            confidence_level="high",
            probabilities={"positive": 0.95, "negative": 0.05},
            inference_time_ms=10.5,
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert "sentiment" in d
        assert "confidence" in d
        assert "probabilities" in d
    
    def test_to_dict_truncates_long_text(self):
        """Test that to_dict truncates long text."""
        long_text = "a" * 200
        result = PredictionResult(
            text=long_text,
            sentiment="positive",
            confidence=0.95,
            confidence_level="high",
            probabilities={"positive": 0.95, "negative": 0.05},
            inference_time_ms=10.5,
        )
        
        d = result.to_dict()
        
        assert len(d["text"]) < len(long_text)
        assert "..." in d["text"]
    
    def test_to_dict_rounds_values(self):
        """Test that to_dict rounds numeric values."""
        result = PredictionResult(
            text="Test",
            sentiment="positive",
            confidence=0.9512345,
            confidence_level="high",
            probabilities={"positive": 0.9512345, "negative": 0.0487655},
            inference_time_ms=10.12345,
        )
        
        d = result.to_dict()
        
        # Check rounding
        assert d["confidence"] == 0.9512
        assert d["inference_time_ms"] == 10.12


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
    
    def test_predict_has_inference_time(self, model, preprocessor, device):
        """Test that prediction includes inference time."""
        predictor = SentimentPredictor(model, preprocessor, device)
        result = predictor.predict("Test movie review")
        
        assert result.inference_time_ms > 0
    
    def test_predict_invalid_input_raises(self, model, preprocessor, device):
        """Test that predict raises error for invalid input."""
        predictor = SentimentPredictor(model, preprocessor, device)
        
        with pytest.raises(ValueError):
            predictor.predict(None)
        
        with pytest.raises(ValueError):
            predictor.predict("")
    
    def test_predict_batch_returns_list(self, model, preprocessor, device, sample_texts):
        """Test that predict_batch returns list of results."""
        predictor = SentimentPredictor(model, preprocessor, device)
        results = predictor.predict_batch(sample_texts)
        
        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
        assert all(isinstance(r, PredictionResult) for r in results)
    
    def test_predict_batch_invalid_input_raises(self, model, preprocessor, device):
        """Test that predict_batch raises for invalid inputs."""
        predictor = SentimentPredictor(model, preprocessor, device)
        
        with pytest.raises(ValueError):
            predictor.predict_batch(["valid", None, "another"])
    

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
        # Confidence might differ slightly due to numerical precision
        assert abs(single_result.confidence - batch_result.confidence) < 0.01
