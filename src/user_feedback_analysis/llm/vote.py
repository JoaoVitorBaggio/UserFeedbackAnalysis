"""Voto majoritário das K inferências de um exemplo.

O protocolo pede K=3 e a moda. Um placar 2–1 tem moda única.
Sem moda única (em K=3, três rótulos diferentes), o desempate é a
primeira inferência entre as empatadas (``first_inference``).

A ordem SAT, DSAT, sem sinal não desempata: num 1–1–1 ela elegeria
SAT sempre. A primeira inferência não prefere classe. O nome da regra
entra no JSON da corrida. A call de 31/10/2026 deixou este desempate
em aberto; esta é a regra do código.
"""

from __future__ import annotations

from collections.abc import Sequence

from user_feedback_analysis.labels import LABELS, Label

TIE_BREAK = "first_inference"
VOTE = "majority"


def majority_vote(votes: Sequence[Label]) -> tuple[Label, str]:
    """Devolve ``(rótulo, decisão)``.

    ``decisão`` é ``majority`` quando há uma moda, ou ``first_inference``
    quando duas ou mais classes empatam no máximo. Nesse caso vence o
    rótulo da primeira inferência que está no empate.
    """
    if not votes:
        raise ValueError("Voto sem inferências.")
    counts = {label: 0 for label in LABELS}
    for label in votes:
        if label not in counts:
            raise ValueError(f"Rótulo {label!r} está fora do voto.")
        counts[label] += 1
    top = max(counts.values())
    modes = [label for label in LABELS if counts[label] == top]
    if len(modes) == 1:
        return modes[0], "majority"
    tied = set(modes)
    for label in votes:
        if label in tied:
            return label, TIE_BREAK
    raise RuntimeError("Desempate sem candidato.")
