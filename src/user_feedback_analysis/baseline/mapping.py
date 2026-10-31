"""Projeção da saída de sentimento para SAT, DSAT e sem sinal.

Hipótese de trabalho do protocolo: positivo → SAT, negativo → DSAT, neutro → sem sinal.
Checkpoint sem neutro não entra: o mapa ficaria a definir e o protocolo recusa
colapsar para duas classes.

O checkpoint padrão é ``cardiffnlp/twitter-roberta-base-sentiment-latest``
(RoBERTa-base, TweetEval / TimeLMs). O card expõe Negative, Neutral e Positive.
A alternativa de três classes, que não é a corrida principal, é
``j-hartmann/sentiment-roberta-large-english-3-classes``.
"""

from __future__ import annotations

from typing import Mapping

from user_feedback_analysis.labels import Label

DEFAULT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ALTERNATIVE_MODEL_ID = "j-hartmann/sentiment-roberta-large-english-3-classes"

SENTIMENT_TO_LABEL: dict[str, Label] = {
    "positive": Label.SAT,
    "negative": Label.DSAT,
    "neutral": Label.NO_SIGNAL,
}

# O modelo antigo do mesmo grupo nomeia as classes como LABEL_0/1/2.
_KNOWN_LABEL_N: dict[str, dict[int, str]] = {
    "cardiffnlp/twitter-roberta-base-sentiment": {
        0: "negative",
        1: "neutral",
        2: "positive",
    }
}


def mapping_dict() -> dict[str, str]:
    return {name: label.value for name, label in SENTIMENT_TO_LABEL.items()}


def sentiment_to_label(sentiment: str) -> Label:
    key = sentiment.strip().casefold()
    try:
        return SENTIMENT_TO_LABEL[key]
    except KeyError as exc:
        raise ValueError(
            f"Sentimento {sentiment!r} está fora de positivo, negativo e neutro. "
            "Sem neutro o mapa fica a definir; o baseline não colapsa para duas classes."
        ) from exc


def validate_id2label(id2label: Mapping[object, object], model_id: str) -> dict[int, str]:
    """Devolve índice → {negative, neutral, positive}. Falha se faltar uma das três."""
    normalized: dict[int, str] = {}
    for key, value in id2label.items():
        normalized[int(key)] = str(value).strip().casefold()
    values = set(normalized.values())
    if values == set(SENTIMENT_TO_LABEL):
        return normalized
    if values <= {"label_0", "label_1", "label_2"} and model_id in _KNOWN_LABEL_N:
        return dict(_KNOWN_LABEL_N[model_id])
    raise ValueError(
        f"O checkpoint {model_id} não expõe positivo, negativo e neutro "
        f"(id2label={dict(id2label)}). Sem neutro o mapa fica a definir; "
        "o baseline não colapsa para duas classes."
    )


def pipeline_label_to_sentiment(raw_label: str, id2label_norm: Mapping[int, str]) -> str:
    name = raw_label.strip().casefold()
    if name in SENTIMENT_TO_LABEL:
        return name
    if name.startswith("label_"):
        idx = int(name.split("_", 1)[1])
        mapped = id2label_norm.get(idx)
        if mapped in SENTIMENT_TO_LABEL:
            return str(mapped)
    raise ValueError(
        f"Saída {raw_label!r} está fora de positivo, negativo e neutro."
    )
