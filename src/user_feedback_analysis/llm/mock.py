"""Resposta local quando não há chave de API.

Não é um LLM e não é o baseline de sentimento. A corrida marca ``dry_run``.
"""

from __future__ import annotations

from user_feedback_analysis.labels import Label

_NEGATIVE = ("wrong", "does not work", "doesn't work", "still broken", "fix it", "failed")
_POSITIVE = ("thank", "perfect", "great", "exactly what")


def mock_label(text: str) -> Label:
    folded = text.casefold()
    if any(cue in folded for cue in _NEGATIVE):
        return Label.DSAT
    if any(cue in folded for cue in _POSITIVE):
        return Label.SAT
    return Label.NO_SIGNAL
