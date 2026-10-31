"""Extrai um único rótulo da resposta do modelo."""

from __future__ import annotations

import re

from user_feedback_analysis.labels import Label, normalize_label

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sem sinal", re.compile(r"\bsem[\s_-]+sinal\b", re.IGNORECASE)),
    ("no-signal", re.compile(r"\bno[\s_-]*signal\b", re.IGNORECASE)),
    ("dsat", re.compile(r"\bdsat\b", re.IGNORECASE)),
    ("sat", re.compile(r"\bsat\b", re.IGNORECASE)),
)


def parse_label(text: str) -> Label:
    stripped = text.strip().strip("`\"'")
    try:
        return normalize_label(stripped)
    except ValueError:
        pass
    found: list[Label] = []
    for alias, pattern in _PATTERNS:
        if not pattern.search(text):
            continue
        label = normalize_label(alias)
        if label not in found:
            found.append(label)
    if len(found) == 1:
        return found[0]
    raise ValueError(f"Resposta do LLM sem um rótulo único: {text!r}.")
