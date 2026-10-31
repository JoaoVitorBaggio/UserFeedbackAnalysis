"""Classes do protocolo: SAT, DSAT e sem sinal."""

from __future__ import annotations

from enum import Enum


class Label(str, Enum):
    SAT = "SAT"
    DSAT = "DSAT"
    NO_SIGNAL = "sem sinal"

    def __str__(self) -> str:
        return self.value


LABELS: tuple[Label, ...] = (Label.SAT, Label.DSAT, Label.NO_SIGNAL)

_ALIASES: dict[str, Label] = {
    "sat": Label.SAT,
    "dsat": Label.DSAT,
    "sem sinal": Label.NO_SIGNAL,
    "sem_sinal": Label.NO_SIGNAL,
    "sem-sinal": Label.NO_SIGNAL,
    "no-signal": Label.NO_SIGNAL,
    "no_signal": Label.NO_SIGNAL,
    "no signal": Label.NO_SIGNAL,
    "nosignal": Label.NO_SIGNAL,
}


def normalize_label(value: Label | str) -> Label:
    """Aceita o enum, o nome canônico e aliases (no-signal → sem sinal)."""
    if isinstance(value, Label):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"Rótulo {value!r} está fora de {{SAT, DSAT, sem sinal}}."
        )
    key = " ".join(value.strip().casefold().split())
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Rótulo {value!r} está fora de {{SAT, DSAT, sem sinal}}."
        ) from exc


def label_from_flags(satisfaction: bool, dissatisfaction: bool) -> Label | None:
    """Mapa fechado em docs/definicao.md.

    SAT: Satisfaction e não Disatisfaction.
    DSAT: Disatisfaction e não Satisfaction.
    sem sinal: os dois falsos.
    Os dois verdadeiros não têm regra: devolve None para o caller descartar.
    """
    if satisfaction and dissatisfaction:
        return None
    if satisfaction:
        return Label.SAT
    if dissatisfaction:
        return Label.DSAT
    return Label.NO_SIGNAL
