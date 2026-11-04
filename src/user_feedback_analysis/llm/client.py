"""Cliente compatível com a API de chat da OpenAI.

Variáveis de ambiente, sem chave no código:

- ``OPENAI_API_KEY``: obrigatória na chamada real.
- ``LLM_MODEL``: obrigatória na chamada real. A decisão de 2026-11-03
  fecha ``openai/gpt-oss-20b`` na Groq. Llama 3.1 8B Instruct foi revogado.
- ``OPENAI_BASE_URL``: opcional no cliente. Padrão
  ``https://api.openai.com/v1``. A corrida fechada usa
  ``https://api.groq.com/openai/v1``.

A call de 31/10/2026 tinha apontado Llama 3.1 8B Instruct, via Groq ou
OpenRouter (``meta-llama/llama-3.1-8b-instruct``). A conta Groq do João
não lista mais esse modelo. O id histórico ``llama-3.1-8b-instant`` já
constava como desligado em 16/08/2026 fora de contrato enterprise. Essa
rota não é a corrida de 2026-11-03.

Sem chave, a corrida cai no dry-run e não chama a rede. Cada exemplo pede
K inferências (padrão 3) e devolve a moda. A temperatura padrão do cliente
é 0.7, para as amostras poderem divergir; 0 continua aceito e não é
exigência do protocolo. O valor vai no corpo da chamada e no JSON.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from user_feedback_analysis.labels import Label
from user_feedback_analysis.llm.mock import mock_label
from user_feedback_analysis.llm.parse import parse_label
from user_feedback_analysis.llm.protocol import (
    PROTOCOL_LLM_BASE_URL,
    PROTOCOL_LLM_MODEL_ID,
    REVOKED_LLM_REQUIREMENT,
)
from user_feedback_analysis.llm.vote import majority_vote
from user_feedback_analysis.records import Example

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DOCUMENTED_LLM_NAME = PROTOCOL_LLM_MODEL_ID
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_LLAMA_31_8B = "meta-llama/llama-3.1-8b-instruct"
GROQ_BASE_URL = PROTOCOL_LLM_BASE_URL
DEFAULT_INFERENCE_K = 3
# Padrão do cliente, não hiperparâmetro medido e não número do protocolo.
# Com temperatura 0 um endpoint determinístico devolve três rótulos iguais
# e o voto majoritário não separa nada. 0.7 deixa as K amostras poderem divergir.
DEFAULT_TEMPERATURE = 0.7

_MISSING_MODEL = (
    "LLM_MODEL ausente. A decisão de 2026-11-03 usa "
    f"LLM_MODEL={PROTOCOL_LLM_MODEL_ID} e "
    f"OPENAI_BASE_URL={PROTOCOL_LLM_BASE_URL}. "
    f"{REVOKED_LLM_REQUIREMENT} foi revogado: a conta Groq do João não lista "
    "mais esse modelo. A rota antiga de OpenRouter, também revogada, era "
    f"LLM_MODEL={OPENROUTER_LLAMA_31_8B} e "
    f"OPENAI_BASE_URL={OPENROUTER_BASE_URL}. "
    "Passe o id pela variável LLM_MODEL. Não há chave no código."
)


def resolve_dry_run(requested: bool | None) -> bool:
    """True pede mock. None usa mock quando falta chave.

    Chave presente e modelo ausente é erro, para não gravar um mock no lugar
    de uma chamada que parecia real.
    """
    if requested is True:
        return True
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    has_model = bool(os.environ.get("LLM_MODEL"))
    if requested is False or has_key:
        if not has_key:
            raise RuntimeError(
                "OPENAI_API_KEY ausente. Exporte a chave para uma chamada real "
                "ou rode com --dry-run."
            )
        if not has_model:
            raise RuntimeError(_MISSING_MODEL)
        return False
    return True


def chat_complete(
    prompt: str,
    *,
    model: str,
    timeout: float = 60.0,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente.")
    base = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "You output only one classification label.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Falha ao chamar o LLM: {exc.reason}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Resposta do LLM sem choices[0].message.content.") from exc
    if not isinstance(content, str):
        raise RuntimeError("O conteúdo da resposta do LLM não é texto.")
    return content


@dataclass(frozen=True)
class VoteResult:
    label: Label
    votes: tuple[Label, ...]
    decision: str


def classify_votes(
    text: str,
    demonstrations: list[Example],
    *,
    dry_run: bool,
    model: str | None,
    max_chars: int | None,
    timeout: float = 60.0,
    k: int = DEFAULT_INFERENCE_K,
    temperature: float = DEFAULT_TEMPERATURE,
    complete: Callable[..., str] | None = None,
) -> VoteResult:
    """K inferências do mesmo prompt. A resposta é a moda dos rótulos parseados.

    Dry-run repete o mock local K vezes e não chama ``complete`` nem a rede.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("K é um inteiro >= 1. O protocolo usa K=3.")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) < 0
    ):
        raise ValueError("temperature é um número >= 0. O protocolo não exige 0.")
    from user_feedback_analysis.llm.prompts import build_prompt, clip_text

    prompt, _ = build_prompt(text, demonstrations, max_chars=max_chars)
    if dry_run:
        shown, _ = clip_text(text, max_chars)
        votes = [mock_label(shown) for _ in range(k)]
    else:
        if not model:
            raise RuntimeError(_MISSING_MODEL)
        sender = complete if complete is not None else chat_complete
        votes = []
        for _ in range(k):
            raw = sender(
                prompt,
                model=model,
                timeout=timeout,
                temperature=float(temperature),
            )
            votes.append(parse_label(raw))
    label, decision = majority_vote(votes)
    return VoteResult(label=label, votes=tuple(votes), decision=decision)


def classify_text(
    text: str,
    demonstrations: list[Example],
    *,
    dry_run: bool,
    model: str | None,
    max_chars: int | None,
    timeout: float = 60.0,
    k: int = DEFAULT_INFERENCE_K,
    temperature: float = DEFAULT_TEMPERATURE,
    complete: Callable[..., str] | None = None,
) -> Label:
    return classify_votes(
        text,
        demonstrations,
        dry_run=dry_run,
        model=model,
        max_chars=max_chars,
        timeout=timeout,
        k=k,
        temperature=temperature,
        complete=complete,
    ).label
