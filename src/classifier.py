"""Misconception classifier wrapper for inference."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class MisconceptionClassifier:
    """Wraps a fine-tuned transformer for misconception classification."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

        # Move to best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def predict(self, question: str, student_response: str) -> dict:
        """Classify a student response.

        Returns:
            dict with keys: label, confidence, all_probs
        """
        text = f"Question: {question}\nStudent answer: {student_response}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        enc.pop("token_type_ids", None)  # DistilBERT doesn't use token type IDs

        with torch.no_grad():
            outputs = self.model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = probs.argmax().item()
        label = self.id2label[pred_idx]
        confidence = probs[pred_idx].item()

        all_probs = {
            self.id2label[i]: round(p.item(), 4)
            for i, p in enumerate(probs)
        }

        return {
            "label": label,
            "confidence": confidence,
            "all_probs": all_probs,
        }
