from __future__ import annotations

from typing import Any

from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipeline = pipeline("sentiment-analysis", model=model_name)

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> dict[str, Any]:
        result = self.pipeline(text, truncation=True)[0]
        return {
            "text": text,
            "label": result["label"],
            "score": round(float(result["score"]), 4),
            "model_name": self.model_name,
        }
