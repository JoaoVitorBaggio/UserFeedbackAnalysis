"""Dataset já partido. Nikolas tokeniza no treino; aqui o item é texto + índice."""

from __future__ import annotations

from typing import Sequence

from user_feedback_analysis.labels import LABELS, Label
from user_feedback_analysis.records import Example

LABEL_TO_ID: dict[Label, int] = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL: dict[int, Label] = {index: label for label, index in LABEL_TO_ID.items()}


class SatisfactionDataset:
    """Três classes, índices estáveis: 0 SAT, 1 DSAT, 2 sem sinal.

    Construa com ``split.train`` ou ``split.val`` de ``make_splits``.
    O teste oficial não entra em early stopping.
    """

    def __init__(self, examples: Sequence[Example]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        if example.label not in LABEL_TO_ID:
            raise ValueError(f"Rótulo fora das três classes: {example.label!r}.")
        return {
            "id": example.example_id,
            "text": example.text,
            "label": LABEL_TO_ID[example.label],
            "label_name": example.label.value,
        }
