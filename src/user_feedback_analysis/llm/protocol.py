"""Decisão de LLM fechada por João em 2026-11-03 (America/Sao_Paulo).

A call de 31/10/2026 tinha fechado Llama 3.1 8B Instruct. Essa exigência
foi revogada: a conta Groq do João não lista mais esse modelo. As
corridas de LLM deste protocolo usam ``openai/gpt-oss-20b`` na Groq e
avaliam 1000 turnos estratificados do teste oficial de 12%.

``official_result`` sai verdadeiro só quando a chamada real casa com
esse arranjo. Dry-run e qualquer desvio ficam falsos, com a lista
``protocol_deviations``. Cardiff e Transformer não usam esta regra:
o harness deles continua com ``official_result`` falso.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from user_feedback_analysis.data.split import PROTOCOL_TEST_FRACTION

PROTOCOL_LLM_DECISION = "2026-11-03"
PROTOCOL_LLM_TIMEZONE = "America/Sao_Paulo"
PROTOCOL_LLM_MODEL_ID = "openai/gpt-oss-20b"
PROTOCOL_LLM_BASE_URL = "https://api.groq.com/openai/v1"
PROTOCOL_LLM_TEST_SUBSAMPLE = 1000
PROTOCOL_FEW_SHOT_K = 3
PROTOCOL_LLM_SEEDS = (0, 1, 2)
PROTOCOL_PER_CLASS = 5000
PROTOCOL_INFERENCE_K = 3
REVOKED_LLM_REQUIREMENT = "Llama 3.1 8B Instruct"

OFFICIAL_RESULT_POLICY = (
    "Decisão de 2026-11-03 (America/Sao_Paulo): official_result é true só na "
    "chamada real com LLM_MODEL=openai/gpt-oss-20b, "
    "OPENAI_BASE_URL=https://api.groq.com/openai/v1, teste oficial em fração 0.12, "
    "--test-subsample 1000, --per-class 5000 --strict-per-class, "
    "--inferences 3, few-shot com k=3, semente 0, 1 ou 2. "
    "true quer dizer que a chamada casa com esse protocolo de LLM. "
    "O JSON continua local, fora do git, e não é sozinho o registro do artigo. "
    "Dry-run e qualquer desvio ficam false, com protocol_deviations. "
    "Isto revoga Llama 3.1 8B Instruct. "
    "Cardiff e Transformer não usam esta regra e seguem com false."
)


def model_requirement_record() -> dict[str, str]:
    return {
        "decision": PROTOCOL_LLM_DECISION,
        "timezone": PROTOCOL_LLM_TIMEZONE,
        "accepted_model_id": PROTOCOL_LLM_MODEL_ID,
        "accepted_base_url": PROTOCOL_LLM_BASE_URL,
        "revoked_model": REVOKED_LLM_REQUIREMENT,
        "revoked_status": "revoked",
    }


def public_base_url(url: str | None, *, fallback: str) -> str:
    """Base HTTP(S) sem usuário, senha, query ou fragmento.

    Uma URL sem esquema, ou que não seja http(s), vira string vazia.
    O valor bruto não volta: pode ser chave colada no lugar da base.
    """
    raw = (url if url else fallback).strip()
    parts = urlsplit(raw)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return ""
    host = parts.hostname.lower()
    netloc = f"{host}:{parts.port}" if parts.port else host
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), netloc, path, "", ""))


def llm_official_result(
    *,
    dry_run: bool,
    model_id: str | None,
    base_url: str | None,
    shot: str,
    few_shot_k: int,
    inferences: int,
    test_subsample: int | None,
    test_fraction: float | None,
    per_class: int,
    strict_per_class: bool,
    seed: int,
    fallback_base_url: str,
) -> tuple[bool, list[str]]:
    """True só no arranjo fechado em 2026-11-03. A lista explica o false."""
    deviations: list[str] = []
    if dry_run:
        deviations.append("dry_run: a chamada não foi à rede.")
    normalized_model = model_id.strip() if isinstance(model_id, str) else None
    if normalized_model != PROTOCOL_LLM_MODEL_ID:
        shown = normalized_model if normalized_model else "(ausente)"
        deviations.append(
            f"model_id {shown} não é {PROTOCOL_LLM_MODEL_ID}, "
            f"o id fechado em {PROTOCOL_LLM_DECISION}. "
            f"{REVOKED_LLM_REQUIREMENT} foi revogado."
        )
    got_base = public_base_url(base_url, fallback=fallback_base_url)
    if got_base != PROTOCOL_LLM_BASE_URL:
        shown_base = got_base if got_base else "(vazia ou inválida)"
        deviations.append(
            f"base_url {shown_base} não é {PROTOCOL_LLM_BASE_URL}."
        )
    if test_subsample != PROTOCOL_LLM_TEST_SUBSAMPLE:
        deviations.append(
            f"test_subsample {test_subsample!r} não é {PROTOCOL_LLM_TEST_SUBSAMPLE}. "
            "A avaliação do LLM é a subamostra estratificada do teste oficial, "
            "não o teste de ~63 mil turnos."
        )
    if not _fraction_is_protocol(test_fraction):
        deviations.append(
            "o teste oficial desta chamada não é a fração 0.12. "
            "A subamostra precisa sair desse hold-out."
        )
    per_class_ok = _is_int(per_class) and per_class == PROTOCOL_PER_CLASS
    if not per_class_ok or strict_per_class is not True:
        deviations.append(
            "treino do protocolo é --per-class 5000 --strict-per-class."
        )
    if not _is_int(inferences) or inferences != PROTOCOL_INFERENCE_K:
        deviations.append("K de inferências do protocolo é 3.")
    if shot == "zero":
        if few_shot_k != 0:
            deviations.append("zero-shot não usa demonstração.")
    elif shot == "few":
        if few_shot_k != PROTOCOL_FEW_SHOT_K:
            deviations.append(
                f"few-shot fechado em k={PROTOCOL_FEW_SHOT_K} ({PROTOCOL_LLM_DECISION}). "
                "Antes o número de demonstrações estava em aberto."
            )
    else:
        deviations.append(f"shot {shot!r} está fora do protocolo.")
    if not _is_int(seed) or seed not in PROTOCOL_LLM_SEEDS:
        deviations.append("sementes do protocolo são 0, 1 e 2.")
    return (len(deviations) == 0, deviations)


def _fraction_is_protocol(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) == float(PROTOCOL_TEST_FRACTION)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
