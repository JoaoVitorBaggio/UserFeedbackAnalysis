"""Checkpoint da corrida de LLM no próprio ``--output``.

Em 2026-11-04 (America/Sao_Paulo) João pediu para a avaliação ao vivo
não perder o que já tinha sido pago na DeepInfra. O JSON final só
nascia no fim da semente. Agora cada exemplo concluído substitui o
arquivo do ``--output`` por um checkpoint (replace atômico). Rodar de
novo o mesmo comando continua desse índice.

O checkpoint não é o relatório e sai com ``official_result`` falso.
O JSON final, escrito quando a semente fecha, é que carrega a regra
de sempre. Retomar não entra nessa regra: não muda modelo, base,
subamostra, K, k nem semente.

Um exemplo conta como concluído quando as K inferências já viraram
um voto. Queda no meio de um exemplo, ou recusa que esgotou as
tentativas, deixa o anterior no disco e repete esse exemplo na
próxima invocação. O arquivo de outra corrida no mesmo caminho é
erro: não é apagado e não é misturado.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from user_feedback_analysis.harness import write_json
from user_feedback_analysis.labels import normalize_label
from user_feedback_analysis.llm.client import VoteResult
from user_feedback_analysis.llm.vote import majority_vote
from user_feedback_analysis.records import Example

CHECKPOINT_KIND = "llm_checkpoint"
CHECKPOINT_VERSION = 1
_FINAL_ARMS = ("llm_zero_shot", "llm_few_shot")


def build_fingerprint(
    *,
    shot: str,
    seed: int,
    few_shot_seed: int | None,
    few_shot_k: int,
    inferences: int,
    temperature: float,
    max_chars: int | None,
    dry_run: bool,
    model_id: str | None,
    base_url: str,
    test_subsample: int | None,
    test_fraction: float | None,
    test_size: int | None,
    per_class: int,
    strict_per_class: bool,
    split_unit: str,
    role: str | None,
    val_size: int | None,
    demonstrations: Sequence[Example],
    evaluated: Sequence[Example],
) -> dict[str, object]:
    """Identidade da corrida, já no formato que o JSON devolve.

    O hash cobre id, rótulo e texto. O ``example_id`` sozinho é
    ``r{índice}``: outro corpus no mesmo índice não pode herdar o voto.
    """
    payload: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "shot": shot,
        "seed": seed,
        "few_shot_seed": few_shot_seed,
        "k": few_shot_k,
        "inference_k": inferences,
        "temperature": float(temperature),
        "max_chars": max_chars,
        "dry_run": dry_run,
        "model_id": model_id,
        "base_url": base_url,
        "test_subsample": test_subsample,
        "test_fraction": test_fraction,
        "test_size": test_size,
        "per_class": per_class,
        "strict_per_class": strict_per_class,
        "split_unit": split_unit,
        "role": role,
        "val_size": val_size,
        "demonstration_ids": [example.example_id for example in demonstrations],
        "evaluated_ids": [example.example_id for example in evaluated],
        "demonstration_digest": _digest(demonstrations),
        "evaluated_digest": _digest(evaluated),
    }
    frozen = json.loads(json.dumps(payload, ensure_ascii=False))
    if not isinstance(frozen, dict):
        raise RuntimeError("Fingerprint não serializou como objeto.")
    return frozen


def write_checkpoint(
    path: Path,
    fingerprint: Mapping[str, object],
    examples: Sequence[Example],
    voted: Sequence[VoteResult],
) -> None:
    """Persiste o prefixo já votado. ``next_index`` é o próximo exemplo."""
    if len(voted) > len(examples):
        raise ValueError("Checkpoint com mais votos do que exemplos.")
    completed = [
        _row(example, result) for example, result in zip(examples, voted)
    ]
    write_json(
        {
            "kind": CHECKPOINT_KIND,
            "version": CHECKPOINT_VERSION,
            "complete": len(voted) == len(examples),
            "next_index": len(voted),
            "n_completed": len(voted),
            "n_expected": len(examples),
            "official_result": False,
            "fingerprint": dict(fingerprint),
            "completed": completed,
            "note": (
                "Checkpoint da corrida de LLM. Não é o JSON final e "
                "official_result fica falso. Retomar é rodar de novo "
                "com o mesmo --output."
            ),
        },
        path,
    )


def load_resume(
    path: Path,
    fingerprint: Mapping[str, object],
    examples: Sequence[Example],
) -> tuple[VoteResult, ...]:
    """Votos já gravados neste caminho. Vazio se o arquivo não existe.

    Não apaga e não reescreve. Arquivo de outra corrida, JSON truncado
    ou voto que não fecha com a moda levantam ``ValueError``.
    """
    if not path.exists():
        return ()
    saved = _read_object(path)
    if saved.get("kind") == CHECKPOINT_KIND:
        return _from_checkpoint(path, saved, fingerprint, examples)
    if _looks_like_final(saved):
        return _from_final(path, saved, fingerprint, examples)
    raise ValueError(
        f"{path} já existe e não é uma corrida de LLM retomável. "
        "Não apaguei o arquivo."
    )


def _from_checkpoint(
    path: Path,
    saved: Mapping[str, object],
    fingerprint: Mapping[str, object],
    examples: Sequence[Example],
) -> tuple[VoteResult, ...]:
    version = _as_int(saved.get("version"), "version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} é um checkpoint versão {version}. "
            f"Este código lê a versão {CHECKPOINT_VERSION}. Não apaguei o arquivo."
        )
    saved_fp = saved.get("fingerprint")
    if not isinstance(saved_fp, dict):
        raise ValueError(f"{path} não tem fingerprint. Não apaguei o arquivo.")
    mismatch = _fingerprint_diff(saved_fp, fingerprint)
    if mismatch is not None:
        raise ValueError(
            f"{path} já existe e não é esta corrida ({mismatch}). "
            "Não apaguei o arquivo."
        )
    completed = saved.get("completed")
    if not isinstance(completed, list):
        raise ValueError(f"{path} não tem a lista completed. Não apaguei o arquivo.")
    next_index = _as_int(saved.get("next_index"), "next_index")
    n_completed = _as_int(saved.get("n_completed"), "n_completed")
    n_expected = _as_int(saved.get("n_expected"), "n_expected")
    if next_index != n_completed or next_index != len(completed):
        raise ValueError(
            f"{path} tem next_index {next_index}, n_completed {n_completed} "
            f"e {len(completed)} linhas. Checkpoint inconsistente. "
            "Não apaguei o arquivo."
        )
    if n_expected != len(examples) or next_index > len(examples):
        raise ValueError(
            f"{path} espera {n_expected} exemplos e tem {next_index} prontos; "
            f"esta corrida tem {len(examples)}. Não apaguei o arquivo."
        )
    if saved.get("complete") is True and next_index != len(examples):
        raise ValueError(
            f"{path} marca complete com {next_index} de {len(examples)}. "
            "Não apaguei o arquivo."
        )
    if saved.get("complete") is False and next_index == len(examples) and examples:
        raise ValueError(
            f"{path} tem todos os exemplos e complete falso. "
            "Não apaguei o arquivo."
        )
    return _rows_to_votes(path, completed, examples, fingerprint)


def _from_final(
    path: Path,
    saved: Mapping[str, object],
    fingerprint: Mapping[str, object],
    examples: Sequence[Example],
) -> tuple[VoteResult, ...]:
    """JSON final desta corrida: todos os exemplos, nenhum a refazer.

    Arquivo novo compara o ``run_fingerprint``. Arquivo anterior ao
    checkpoint não tem hash do texto; a identidade fica nos botões e
    nos ids gravados. Os dois recusam mistura e não são apagados.
    """
    saved_fp = saved.get("run_fingerprint")
    if isinstance(saved_fp, dict):
        mismatch = _fingerprint_diff(saved_fp, fingerprint)
        if mismatch is not None:
            raise ValueError(
                f"{path} já existe e não é esta corrida ({mismatch}). "
                "Não apaguei o arquivo."
            )
    else:
        _legacy_matches(path, saved, fingerprint)
    predictions = saved.get("predictions")
    votes = saved.get("inference_votes")
    if not isinstance(predictions, list) or not isinstance(votes, list):
        raise ValueError(f"{path} não tem predictions e inference_votes.")
    if len(predictions) != len(examples) or len(votes) != len(examples):
        raise ValueError(
            f"{path} tem {len(predictions)} predições para {len(examples)} exemplos. "
            "JSON final incompleto não retoma o meio. Não apaguei o arquivo."
        )
    rows: list[object] = []
    for pred, vote in zip(predictions, votes):
        if not isinstance(pred, dict):
            raise ValueError(f"{path} tem uma predição que não é objeto.")
        rows.append(
            {
                "id": pred.get("id"),
                "gold": pred.get("gold"),
                "pred": pred.get("pred"),
                "votes": vote,
            }
        )
    return _rows_to_votes(path, rows, examples, fingerprint)


def _legacy_matches(
    path: Path,
    saved: Mapping[str, object],
    fingerprint: Mapping[str, object],
) -> None:
    split = saved.get("split")
    if not isinstance(split, dict):
        raise ValueError(f"{path} não tem split. Não apaguei o arquivo.")
    problems: list[str] = []

    def check(name: str, got: object, key: str) -> None:
        if _canonical(got) != _canonical(fingerprint.get(key)):
            problems.append(name)

    check("shot", saved.get("shot"), "shot")
    check("seed", split.get("seed"), "seed")
    check("k", saved.get("k"), "k")
    check("few_shot_seed", saved.get("few_shot_seed"), "few_shot_seed")
    check("inference_k", saved.get("inference_k"), "inference_k")
    check("temperature", saved.get("temperature"), "temperature")
    check("max_chars", saved.get("max_chars"), "max_chars")
    check("dry_run", saved.get("dry_run"), "dry_run")
    check("model_id", saved.get("model_id"), "model_id")
    check("base_url", saved.get("base_url"), "base_url")
    check("demonstration_ids", saved.get("demonstration_ids"), "demonstration_ids")
    subsample = saved.get("test_subsample")
    if isinstance(subsample, dict):
        check("test_subsample", subsample.get("requested"), "test_subsample")
    predictions = saved.get("predictions")
    if isinstance(predictions, list):
        ids = [
            row.get("id") if isinstance(row, dict) else None for row in predictions
        ]
        if _canonical(ids) != _canonical(fingerprint.get("evaluated_ids")):
            problems.append("evaluated_ids")
    else:
        problems.append("predictions")
    if problems:
        raise ValueError(
            f"{path} já existe e não é esta corrida ({', '.join(problems)}). "
            "Não apaguei o arquivo."
        )


def _rows_to_votes(
    path: Path,
    rows: Sequence[object],
    examples: Sequence[Example],
    fingerprint: Mapping[str, object],
) -> tuple[VoteResult, ...]:
    inference_k = fingerprint.get("inference_k")
    if isinstance(inference_k, bool) or not isinstance(inference_k, int) or inference_k < 1:
        raise ValueError(f"{path} não tem inference_k válido.")
    voted: list[VoteResult] = []
    for example, row in zip(examples, rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} tem uma linha que não é objeto.")
        example_id = example.example_id
        if row.get("id") != example_id:
            raise ValueError(
                f"{path} tem {row.get('id')!r} onde esta corrida espera {example_id!r}. "
                "Não apaguei o arquivo."
            )
        if row.get("gold") != example.label.value:
            raise ValueError(
                f"{path} guarda ouro {row.get('gold')!r} em {example_id}, "
                f"e o corpus agora tem {example.label.value!r}. Não apaguei o arquivo."
            )
        raw_votes = row.get("votes")
        if not isinstance(raw_votes, list) or len(raw_votes) != inference_k:
            raise ValueError(
                f"{path} não tem {inference_k} votos em {example_id}. "
                "Não apaguei o arquivo."
            )
        try:
            labels = [normalize_label(item) for item in raw_votes]
        except ValueError as exc:
            raise ValueError(
                f"{path}: voto inválido em {example_id}: {exc}"
            ) from exc
        label, decision = majority_vote(labels)
        pred = row.get("pred")
        if pred != label.value:
            raise ValueError(
                f"{path} diz {pred!r} em {example_id} e o voto recalculado "
                f"é {label.value!r}. Não apaguei o arquivo."
            )
        stored = row.get("decision")
        if stored is not None and stored != decision:
            raise ValueError(
                f"{path} diz decisão {stored!r} em {example_id} e o recálculo "
                f"é {decision!r}. Não apaguei o arquivo."
            )
        voted.append(VoteResult(label=label, votes=tuple(labels), decision=decision))
    return tuple(voted)


def _row(example: Example, result: VoteResult) -> dict[str, object]:
    return {
        "id": example.example_id,
        "gold": example.label.value,
        "pred": result.label.value,
        "votes": [label.value for label in result.votes],
        "decision": result.decision,
    }


def _digest(examples: Sequence[Example]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.example_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(example.label.value.encode("utf-8"))
        digest.update(b"\0")
        digest.update(example.text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, object]:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} existe, mas o JSON não fecha. "
            "Não retomo por cima de arquivo truncado."
        ) from exc
    except OSError:
        raise
    if not isinstance(saved, dict):
        raise ValueError(f"{path} não é um objeto JSON. Não apaguei o arquivo.")
    return saved


def _looks_like_final(saved: Mapping[str, object]) -> bool:
    return saved.get("arm") in _FINAL_ARMS and isinstance(saved.get("predictions"), list)


def _fingerprint_diff(
    saved: Mapping[str, object],
    current: Mapping[str, object],
) -> str | None:
    saved_c = _canonical(saved)
    current_c = _canonical(dict(current))
    if saved_c == current_c:
        return None
    keys = list(dict.fromkeys([*current.keys(), *saved.keys()]))
    for key in keys:
        if _canonical(saved.get(key)) == _canonical(current.get(key)):
            continue
        if key in {"evaluated_ids", "demonstration_ids", "evaluated_digest", "demonstration_digest"}:
            return f"{key} não bate com esta corrida"
        return (
            f"{key}: o arquivo tem {saved.get(key)!r} e esta corrida tem "
            f"{current.get(key)!r}"
        )
    return "fingerprint não bate"


def _canonical(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} não é inteiro.")
    return value
