"""Baseline de sentimento genérico pré-treinado."""

from user_feedback_analysis.baseline.mapping import (
    DEFAULT_MODEL_ID,
    mapping_dict,
    sentiment_to_label,
)

__all__ = ["DEFAULT_MODEL_ID", "mapping_dict", "sentiment_to_label"]
