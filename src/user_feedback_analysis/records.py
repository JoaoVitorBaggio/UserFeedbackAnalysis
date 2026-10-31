"""Linhas do ``sat_dsat_annotation`` e exemplos já rotulados."""

from __future__ import annotations

from dataclasses import dataclass

from user_feedback_analysis.labels import Label


@dataclass(frozen=True)
class RawRow:
    source_index: int
    conversation_id: int
    turn_id: int | None
    utterance_id: int | None
    role: str
    content: str | None
    satisfaction: bool | None
    dissatisfaction: bool | None
    state: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class LoadedCorpus:
    rows: list[RawRow]
    conversation_note: str
    source: str
    subset_limit: int | None = None


@dataclass(frozen=True)
class Example:
    example_id: str
    conversation_id: int
    text: str
    label: Label
    role: str
    source_index: int
    turn_id: int | None = None
    utterance_id: int | None = None
    state: str | None = None
    domain: str | None = None
