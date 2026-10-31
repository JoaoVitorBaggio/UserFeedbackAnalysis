"""Léxico mínimo para smoke sem baixar o checkpoint.

Não é o baseline pré-treinado e não produz resultado experimental.
"""

from __future__ import annotations

from user_feedback_analysis.baseline.mapping import sentiment_to_label
from user_feedback_analysis.labels import Label

_NEGATIVE = (
    "wrong",
    "does not work",
    "doesn't work",
    "still broken",
    "fix it",
    "failed",
)
_POSITIVE = (
    "thank",
    "perfect",
    "great",
    "exactly what",
)


def lexicon_sentiment(text: str) -> str:
    folded = text.casefold()
    if any(cue in folded for cue in _NEGATIVE):
        return "negative"
    if any(cue in folded for cue in _POSITIVE):
        return "positive"
    return "neutral"


class LexiconMockClassifier:
    """Classificador injetável. Passa pela mesma projeção do baseline real."""

    kind = "mock-lexicon"

    def predict(self, texts: list[str]) -> list[Label]:
        return [sentiment_to_label(lexicon_sentiment(text)) for text in texts]
