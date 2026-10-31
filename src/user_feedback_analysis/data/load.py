"""Leitor do ``sat_dsat_annotation``.

O arquivo oficial fica fora do git (ODC-By, Microsoft). Este módulo aceita um
JSONL local no schema do card ou o dataset ``microsoft/WildFeedback`` via
``datasets``. O card (consulta em out/2026) lista 1.062.105 linhas; isso é
número do card, não uma contagem feita aqui.

Colunas usadas, com a grafia do arquivo: ``Role``, ``Content``, ``Satisfaction``,
``Disatisfaction``, ``UtterranceId``, ``TurnId``. ``Dissatisfaction`` (grafia
corrigida) também é aceita em JSONL local.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from user_feedback_analysis.records import LoadedCorpus, RawRow

DATASET_ID = "microsoft/WildFeedback"
DATASET_CONFIG = "sat_dsat_annotation"

_CONTENT_KEYS = ("Content", "content", "text")
_ROLE_KEYS = ("Role", "role")
_SAT_KEYS = ("Satisfaction", "satisfaction")
_DSAT_KEYS = (
    "Disatisfaction",
    "Dissatisfaction",
    "disatisfaction",
    "dissatisfaction",
)
_UTT_KEYS = ("UtterranceId", "UtteranceId", "utterance_id")
_TURN_KEYS = ("TurnId", "turn_id")
_STATE_KEYS = ("State", "state")
_DOMAIN_KEYS = ("Domain", "domain")


def _lookup(mapping: Mapping[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _python_scalar(value: object) -> object:
    """numpy.int64 / numpy.bool_ expõem ``item()`` e não herdam de int/bool no NumPy 2."""
    item = getattr(value, "item", None)
    if item is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    try:
        return item()
    except (TypeError, ValueError):
        return value


def as_bool(value: object) -> bool | None:
    if value is None:
        return None
    value = _python_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().casefold()
        if key in {"true", "1", "yes"}:
            return True
        if key in {"false", "0", "no"}:
            return False
    raise ValueError(f"Booleano inválido: {value!r}.")


def as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    value = _python_scalar(value)
    if isinstance(value, bool):
        raise ValueError(f"Inteiro inválido: {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ValueError(f"Inteiro inválido: {value!r}.")


def as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def assign_conversation_ids(
    utterance_ids: Sequence[int | None],
) -> tuple[list[int], str]:
    """Infere conversa pela ordem do arquivo.

    Não há chave de conversa no card. Uma conversa nova começa quando
    ``UtterranceId`` volta a 0 ou diminui. Linha sem id vira conversa própria.
    A unidade do split continua em aberto no protocolo; o padrão do código é
    não partir essa conversa inferida entre teste e treino.
    """
    conv_ids: list[int] = []
    conv = 0
    prev: int | None = None
    missing = 0
    start_new = False
    for utterance_id in utterance_ids:
        if utterance_id is None:
            missing += 1
            if conv_ids:
                conv += 1
            conv_ids.append(conv)
            prev = None
            start_new = True
            continue
        if start_new or (prev is not None and (utterance_id == 0 or utterance_id < prev)):
            if conv_ids:
                conv += 1
            start_new = False
        conv_ids.append(conv)
        prev = utterance_id
    note = (
        "conversation_id inferido pela ordem do arquivo e pelo reset de UtterranceId. "
        "O card não traz chave de conversa. A unidade do split segue em aberto no "
        "protocolo; o padrão do código é não partir a conversa inferida."
    )
    if missing:
        note += f" {missing} linhas sem UtterranceId ficaram em conversas próprias."
    return conv_ids, note


def corpus_from_mappings(
    mappings: Sequence[Mapping[str, object]],
    *,
    source: str,
    subset_limit: int | None = None,
) -> LoadedCorpus:
    utterance_ids = [
        as_optional_int(_lookup(mapping, _UTT_KEYS)) for mapping in mappings
    ]
    conv_ids, note = assign_conversation_ids(utterance_ids)
    rows: list[RawRow] = []
    for source_index, (mapping, conversation_id, utterance_id) in enumerate(
        zip(mappings, conv_ids, utterance_ids)
    ):
        content = as_optional_str(_lookup(mapping, _CONTENT_KEYS))
        role = as_optional_str(_lookup(mapping, _ROLE_KEYS)) or ""
        rows.append(
            RawRow(
                source_index=source_index,
                conversation_id=conversation_id,
                turn_id=as_optional_int(_lookup(mapping, _TURN_KEYS)),
                utterance_id=utterance_id,
                role=role,
                content=content,
                satisfaction=as_bool(_lookup(mapping, _SAT_KEYS)),
                dissatisfaction=as_bool(_lookup(mapping, _DSAT_KEYS)),
                state=as_optional_str(_lookup(mapping, _STATE_KEYS)),
                domain=as_optional_str(_lookup(mapping, _DOMAIN_KEYS)),
            )
        )
    return LoadedCorpus(
        rows=rows,
        conversation_note=note,
        source=source,
        subset_limit=subset_limit,
    )


def load_jsonl(
    path: Path | str,
    *,
    limit: int | None = None,
    source: str | None = None,
) -> LoadedCorpus:
    file_path = Path(path)
    mappings: list[dict[str, object]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Linha JSONL não é objeto: {line[:80]!r}.")
            mappings.append(payload)
            if limit is not None and len(mappings) >= limit:
                break
    return corpus_from_mappings(
        mappings,
        source=source or str(file_path),
        subset_limit=limit,
    )


def load_wildfeedback(
    *,
    split: str = "train",
    limit: int | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> LoadedCorpus:
    """Carrega ``microsoft/WildFeedback``, config ``sat_dsat_annotation``.

    ``limit`` materializa só um prefixo (streaming). Prefixo não é a
    distribuição real e não serve como hold-out do experimento.
    Sem ``limit``, materializa o split inteiro. O card lista cerca de um
    milhão de linhas; o pico de RAM não foi medido aqui.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Para ler microsoft/WildFeedback instale requirements.txt e tenha "
            "acesso ao Hugging Face Hub. O arquivo não entra no git. "
            "Um JSONL local entra por load_jsonl."
        ) from exc

    kwargs: dict[str, object] = {
        "path": DATASET_ID,
        "name": DATASET_CONFIG,
        "split": split,
    }
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if limit is not None:
        if limit < 1:
            raise ValueError("limit é um inteiro >= 1.")
        kwargs["streaming"] = True
        dataset = load_dataset(**kwargs)
        mappings = []
        for row in dataset:
            mappings.append(dict(row))
            if len(mappings) >= limit:
                break
    else:
        dataset = load_dataset(**kwargs)
        mappings = [dict(row) for row in dataset]

    source = f"{DATASET_ID}:{DATASET_CONFIG}"
    if revision:
        source += f"@{revision}"
    return corpus_from_mappings(mappings, source=source, subset_limit=limit)


def row_to_mapping(row: RawRow) -> dict[str, object]:
    return {
        "TurnId": row.turn_id,
        "UtterranceId": row.utterance_id,
        "Role": row.role,
        "Content": row.content,
        "Satisfaction": row.satisfaction,
        "Disatisfaction": row.dissatisfaction,
        "State": row.state,
        "Domain": row.domain,
    }


def write_jsonl(corpus: LoadedCorpus, path: Path | str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in corpus.rows:
            handle.write(json.dumps(row_to_mapping(row), ensure_ascii=False) + "\n")
