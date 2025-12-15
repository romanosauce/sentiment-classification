"""
Inference module for sentiment classification.

Provides a API for making predictions and converting
raw model outputs to user-friendly format.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch

from .model import SentimentClassifier
from .preprocessing import TextPreprocessor

logger = logging.getLogger("sentiment_classifier")


SENTIMENT_LABELS = {0: "negative", 1: "positive"}
CONFIDENCE_THRESHOLDS = {"high": 0.8, "medium": 0.6, "low": 0.0}


@dataclass
class PredictionResult:
    """Result of a sentiment prediction."""
    
    text: str
    sentiment: str
    confidence: float
    confidence_level: str
    probabilities: dict[str, float]
    inference_time_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "sentiment": self.sentiment,
            "confidence": round(self.confidence, 4),
            "confidence_level": self.confidence_level,
            "probabilities": {
                k: round(v, 4) for k, v in self.probabilities.items()
            },
            "inference_time_ms": round(self.inference_time_ms, 2),
        }


def get_confidence_level(confidence: float) -> str:
    """
    Get confidence level string from confidence score.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Confidence level: 'high', 'medium', or 'low'
    """
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"


def convert_logits_to_prediction(
    logits: torch.Tensor,
) -> tuple[int, float, dict[str, float]]:
    """
    Convert raw model logits to prediction.
    
    Args:
        logits: Raw model output logits [batch_size, num_classes]
        
    Returns:
        Tuple of (predicted_class, confidence, probabilities_dict)
    """
    probabilities = torch.softmax(logits, dim=-1)
    
    confidence, predicted_class = torch.max(probabilities, dim=-1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    
    prob_dict = {
        SENTIMENT_LABELS[i]: probabilities[0, i].item()
        for i in range(probabilities.shape[-1])
    }
    
    return predicted_class, confidence, prob_dict


def validate_prediction_input(text: Any) -> tuple[bool, str]:
    """
    Validate input for prediction.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if text is None:
        return False, "Input text cannot be None"
    
    if not isinstance(text, str):
        return False, f"Input must be string, got {type(text).__name__}"
    
    text = text.strip()
    if len(text) == 0:
        return False, "Input text cannot be empty"
    
    if len(text) > 50000:
        return False, "Input text exceeds maximum length (50000 characters)"
    
    return True, ""


class SentimentPredictor:
    """
    High-level predictor class for sentiment classification.
    
    Provides a clean API for making predictions with automatic
    preprocessing and output formatting.
    """
    
    def __init__(
        self,
        model: SentimentClassifier,
        preprocessor: TextPreprocessor,
        device: torch.device | None = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            model: Trained sentiment classifier
            preprocessor: Fitted text preprocessor
            device: Device to run inference on
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or torch.device("cpu")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> PredictionResult:
        """
        Make prediction for a single text.
        
        Args:
            text: Input text
            
        Returns:
            PredictionResult with sentiment and confidence
            
        Raises:
            ValueError: If input validation fails
        """
        is_valid, error = validate_prediction_input(text)
        if not is_valid:
            raise ValueError(error)
        
        start_time = time.time()
        
        sequence = self.preprocessor.transform(text)
        
        padded = self.preprocessor.pad_sequences([sequence])
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids)
        
        predicted_class, confidence, probabilities = convert_logits_to_prediction(logits)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return PredictionResult(
            text=text,
            sentiment=SENTIMENT_LABELS[predicted_class],
            confidence=confidence,
            confidence_level=get_confidence_level(confidence),
            probabilities=probabilities,
            inference_time_ms=inference_time,
        )
    
    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """
        Make predictions for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of PredictionResults
        """
        results = []
        
        for i, text in enumerate(texts):
            is_valid, error = validate_prediction_input(text)
            if not is_valid:
                raise ValueError(f"Invalid input at index {i}: {error}")
        
        start_time = time.time()
        
        sequences = [self.preprocessor.transform(text) for text in texts]
        padded = self.preprocessor.pad_sequences(sequences)
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids)
        
        total_time = (time.time() - start_time) * 1000
        per_sample_time = total_time / len(texts)
        
        probabilities = torch.softmax(logits, dim=-1)
        
        for i, text in enumerate(texts):
            confidence, predicted_class = torch.max(probabilities[i], dim=-1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            
            prob_dict = {
                SENTIMENT_LABELS[j]: probabilities[i, j].item()
                for j in range(probabilities.shape[-1])
            }
            
            results.append(PredictionResult(
                text=text,
                sentiment=SENTIMENT_LABELS[predicted_class],
                confidence=confidence,
                confidence_level=get_confidence_level(confidence),
                probabilities=prob_dict,
                inference_time_ms=per_sample_time,
            ))
        
        return results
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: torch.device | None = None,
    ) -> "SentimentPredictor":
        """
        Load predictor from saved model.
        
        Args:
            model_path: Path to saved model directory
            device: Device to run on
            
        Returns:
            Loaded SentimentPredictor
        """
        model = SentimentClassifier.from_pretrained(model_path, device)
        preprocessor = TextPreprocessor.load(model_path)
        
        return cls(model, preprocessor, device)
