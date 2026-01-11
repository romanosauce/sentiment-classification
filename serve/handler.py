"""
Custom TorchServe handler for sentiment classification.
"""

import html
import json
import logging
import os
import re

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class SentimentHandler(BaseHandler):
    """Custom handler for sentiment classification model."""

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.word2idx = {}
        self.max_seq_length = 256
        self.pad_idx = 0
        self.unk_idx = 1
        self.index_to_name = {0: "negative", 1: "positive"}

    def initialize(self, context):
        """Initialize model and preprocessor."""
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        model_pt_path = os.path.join(model_dir, context.manifest["model"]["serializedFile"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()

        # Load preprocessor config
        with open(os.path.join(model_dir, "preprocessor.json")) as f:
            config = json.load(f)
        self.word2idx = config.get("word2idx", {})
        self.max_seq_length = config.get("max_seq_length", 256)

        # Load label mapping
        with open(os.path.join(model_dir, "index_to_name.json")) as f:
            mapping = json.load(f)
        self.index_to_name = {int(k): v for k, v in mapping.items()}

        self.initialized = True
        logger.info(f"Initialized with vocab size {len(self.word2idx)}")

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

    def tokenize(self, text: str) -> list:
        """Tokenize text into padded indices."""
        tokens = self.clean_text(text).split()
        indices = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        # Pad/truncate
        if len(indices) >= self.max_seq_length:
            return indices[:self.max_seq_length]
        return indices + [self.pad_idx] * (self.max_seq_length - len(indices))

    def _extract_text(self, row) -> str:
        """Extract text from various input formats."""
        # Handle bytes
        if isinstance(row, bytes):
            row = row.decode("utf-8")

        # Handle dict with body
        if isinstance(row, dict):
            row = row.get("body") or row.get("data") or row.get("text") or row

        # Handle bytes again (from body)
        if isinstance(row, bytes):
            row = row.decode("utf-8")

        # Try parse JSON string
        if isinstance(row, str):
            try:
                parsed = json.loads(row)
                if isinstance(parsed, dict):
                    return parsed.get("text", parsed.get("data", ""))
                return str(parsed)
            except (json.JSONDecodeError, TypeError):
                return row

        if isinstance(row, dict):
            return row.get("text", row.get("data", ""))

        return str(row) if row else ""

    def preprocess(self, data):
        """Preprocess input data to tensor."""
        sequences = [self.tokenize(self._extract_text(row)) for row in data]
        return torch.tensor(sequences, dtype=torch.long, device=self.device)

    def inference(self, input_tensor):
        """Run model inference."""
        with torch.no_grad():
            return self.model(input_tensor)

    def postprocess(self, inference_output):
        """Convert logits to predictions."""
        probs = torch.softmax(inference_output, dim=-1)
        confidence, predictions = torch.max(probs, dim=-1)

        return [
            {
                "prediction": self.index_to_name[pred.item()],
                "confidence": round(conf.item(), 4),
                "probabilities": {
                    self.index_to_name[j]: round(p, 4)
                    for j, p in enumerate(probs[i].tolist())
                }
            }
            for i, (pred, conf) in enumerate(zip(predictions, confidence))
        ]

    def handle(self, data, context):
        """Main entry point for TorchServe."""
        if not self.initialized:
            self.initialize(context)

        input_tensor = self.preprocess(data)
        output = self.inference(input_tensor)
        return self.postprocess(output)
