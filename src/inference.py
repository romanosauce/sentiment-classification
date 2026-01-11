"""
Inference module for sentiment classification.
"""

import time
from dataclasses import dataclass

import torch

from .model import SentimentClassifier
from .preprocessing import TextPreprocessor

SENTIMENT_LABELS = {0: "negative", 1: "positive"}


@dataclass
class PredictionResult:
    """Result of a sentiment prediction."""
    text: str
    sentiment: str
    confidence: float
    confidence_level: str
    probabilities: dict[str, float]
    inference_time_ms: float


class SentimentPredictor:
    """High-level predictor class for sentiment classification."""

    def __init__(self, model: SentimentClassifier, preprocessor: TextPreprocessor, device: torch.device = None):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> PredictionResult:
        """Make prediction for a single text."""
        start_time = time.time()

        sequence = self.preprocessor.transform(text)
        padded = self.preprocessor.pad_sequences([sequence])
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids)

        probs = torch.softmax(logits, dim=-1)
        confidence, pred_class = torch.max(probs, dim=-1)
        pred_class, confidence = pred_class.item(), confidence.item()

        return PredictionResult(
            text=text,
            sentiment=SENTIMENT_LABELS[pred_class],
            confidence=confidence,
            confidence_level="high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low",
            probabilities={SENTIMENT_LABELS[i]: probs[0, i].item() for i in range(probs.shape[-1])},
            inference_time_ms=(time.time() - start_time) * 1000,
        )

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Make predictions for multiple texts."""
        start_time = time.time()

        sequences = [self.preprocessor.transform(t) for t in texts]
        padded = self.preprocessor.pad_sequences(sequences)
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids)

        probs = torch.softmax(logits, dim=-1)
        total_time = (time.time() - start_time) * 1000

        results = []
        for i, text in enumerate(texts):
            conf, pred = torch.max(probs[i], dim=-1)
            pred, conf = pred.item(), conf.item()
            results.append(PredictionResult(
                text=text,
                sentiment=SENTIMENT_LABELS[pred],
                confidence=conf,
                confidence_level="high" if conf >= 0.8 else "medium" if conf >= 0.6 else "low",
                probabilities={SENTIMENT_LABELS[j]: probs[i, j].item() for j in range(probs.shape[-1])},
                inference_time_ms=total_time / len(texts),
            ))
        return results

    @classmethod
    def from_pretrained(cls, model_path: str, device: torch.device = None) -> "SentimentPredictor":
        """Load predictor from saved model."""
        model = SentimentClassifier.from_pretrained(model_path, device)
        preprocessor = TextPreprocessor.load(model_path)
        return cls(model, preprocessor, device)
