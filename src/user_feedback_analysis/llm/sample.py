"""Exemplos de few-shot saem só do treino.

O protocolo: nenhum braço vê o teste no few-shot. ``k`` está a definir.
"""

from __future__ import annotations

import random
from typing import Sequence

from user_feedback_analysis.labels import LABELS
from user_feedback_analysis.records import Example


def sample_demonstrations(
    train: Sequence[Example],
    k: int,
    seed: int,
) -> list[Example]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k do few-shot é um inteiro >= 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("A semente do few-shot é um inteiro >= 0.")
    if k > len(train):
        raise ValueError(f"k={k} é maior que o treino ({len(train)}).")
    rng = random.Random(seed)
    pools: dict[object, list[Example]] = {label: [] for label in LABELS}
    for example in train:
        pools[example.label].append(example)
    for label in LABELS:
        rng.shuffle(pools[label])
    cursors = {label: 0 for label in LABELS}
    picked: list[Example] = []
    while len(picked) < k:
        moved = False
        for label in LABELS:
            cursor = cursors[label]
            pool = pools[label]
            if cursor < len(pool):
                picked.append(pool[cursor])
                cursors[label] = cursor + 1
                moved = True
                if len(picked) == k:
                    break
        if not moved:
            raise ValueError("O treino não tem exemplos suficientes para o few-shot.")
    return picked
