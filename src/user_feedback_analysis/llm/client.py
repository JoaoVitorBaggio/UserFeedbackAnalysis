"""Cliente compatível com a API de chat da OpenAI.

Variáveis de ambiente, sem chave no código:

- ``OPENAI_API_KEY``: obrigatória na chamada real. Na corrida fechada
  é a chave da DeepInfra. Não entra no repositório.
- ``LLM_MODEL``: obrigatória na chamada real. A decisão de 2026-11-03
  fecha ``openai/gpt-oss-20b``. Llama 3.1 8B Instruct foi revogado.
- ``OPENAI_BASE_URL``: opcional no cliente. Padrão
  ``https://api.openai.com/v1``. Desde 2026-11-04 a corrida fechada usa
  ``https://api.deepinfra.com/v1/openai``. A base da Groq
  ``https://api.groq.com/openai/v1`` foi revogada para estas corridas.

A call de 31/10/2026 tinha apontado Llama 3.1 8B Instruct, via Groq ou
OpenRouter (``meta-llama/llama-3.1-8b-instruct``). A conta Groq do João
não lista mais esse modelo. O id histórico ``llama-3.1-8b-instant`` já
constava como desligado em 16/08/2026 fora de contrato enterprise. Essa
rota não é a corrida de 2026-11-03. Em 2026-11-04 a própria base da Groq
saiu do protocolo: a cota gratuita diária tinha esgotado.

Sem chave, a corrida cai no dry-run e não chama a rede. Cada exemplo pede
K inferências (padrão 3) e devolve a moda. A temperatura padrão do cliente
é 0.7, para as amostras poderem divergir; 0 continua aceito e não é
exigência do protocolo. O valor vai no corpo da chamada e no JSON.

A chamada real manda ``User-Agent`` estável. O padrão do urllib
(``Python-urllib/…``) recebia HTTP 403 da Cloudflare na frente da Groq,
com corpo ``error code: 1010``. O cabeçalho continua em toda chamada,
inclusive na DeepInfra. HTTP 429, HTTP 5xx e falha transitória
de rede repetem no máximo ``LLM_MAX_ATTEMPTS`` vezes. A espera segue
``try again in Xs`` (ou ``ms``) e o header ``Retry-After`` quando vêm
no erro; sem pista, dobra a partir de 1s. Piso e teto cortam a espera,
então a corrida não fica presa numa repetição infinita nem numa pista
de horas.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message

from user_feedback_analysis.labels import Label
from user_feedback_analysis.llm.mock import mock_label
from user_feedback_analysis.llm.parse import parse_label
from user_feedback_analysis.llm.protocol import (
    PROTOCOL_LLM_BACKEND_DECISION,
    PROTOCOL_LLM_BASE_URL,
    PROTOCOL_LLM_MODEL_ID,
    REVOKED_LLM_BASE_URL,
    REVOKED_LLM_REQUIREMENT,
)
from user_feedback_analysis.llm.vote import majority_vote
from user_feedback_analysis.records import Example

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DOCUMENTED_LLM_NAME = PROTOCOL_LLM_MODEL_ID
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_LLAMA_31_8B = "meta-llama/llama-3.1-8b-instruct"
# Base da Groq, revogada para estas corridas em 2026-11-04.
# Não é a base que official_result aceita.
GROQ_BASE_URL = REVOKED_LLM_BASE_URL
DEFAULT_INFERENCE_K = 3
# Padrão do cliente, não hiperparâmetro medido e não número do protocolo.
# Com temperatura 0 um endpoint determinístico devolve três rótulos iguais
# e o voto majoritário não separa nada. 0.7 deixa as K amostras poderem divergir.
DEFAULT_TEMPERATURE = 0.7
# Cloudflare 1010 bloqueava o User-Agent do urllib na Groq. Este passa
# daquela borda e ainda identifica o cliente. Conferido contra api.groq.com
# quando a Groq era o backend: o padrão tomava 403 "error code: 1010";
# este chegava no 401 da API. O cabeçalho permanece na chamada à DeepInfra.
LLM_USER_AGENT = "Mozilla/5.0 (compatible; UserFeedbackAnalysis/llm-client)"
# Uma tentativa inicial mais as repetições. Finito de propósito.
LLM_MAX_ATTEMPTS = 8
_RETRY_BASE_S = 1.0
_RETRY_CAP_S = 120.0
_RETRY_FLOOR_S = 0.05
_TRY_AGAIN_RE = re.compile(
    r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s)\b",
    re.IGNORECASE,
)
_TRANSIENT_REASON_MARKERS = (
    "timed out",
    "timeout",
    "temporarily unavailable",
    "temporary failure",
    "connection reset",
    "connection refused",
    "connection aborted",
    "network is unreachable",
    "name or service not known",
    "nodename nor servname",
)

_MISSING_MODEL = (
    "LLM_MODEL ausente. A decisão de 2026-11-03 usa "
    f"LLM_MODEL={PROTOCOL_LLM_MODEL_ID}. "
    f"Desde {PROTOCOL_LLM_BACKEND_DECISION} a base aceita é "
    f"OPENAI_BASE_URL={PROTOCOL_LLM_BASE_URL}. "
    f"{REVOKED_LLM_REQUIREMENT} foi revogado: a conta Groq do João não lista "
    "mais esse modelo. A base da Groq "
    f"({GROQ_BASE_URL}) também foi revogada em {PROTOCOL_LLM_BACKEND_DECISION}. "
    "A rota antiga de OpenRouter para o Llama, também revogada, era "
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


def _bounded_delay(seconds: float) -> float:
    if seconds < _RETRY_FLOOR_S:
        return _RETRY_FLOOR_S
    if seconds > _RETRY_CAP_S:
        return _RETRY_CAP_S
    return seconds


def _parse_try_again_seconds(detail: str) -> float | None:
    """Segundos em ``try again in 1.5s`` ou ``try again in 250ms``.

    ``200ms`` não pode virar 200 segundos: a unidade entra no casamento.
    """
    match = _TRY_AGAIN_RE.search(detail)
    if match is None:
        return None
    value = float(match.group(1))
    if match.group(2).lower() == "ms":
        value /= 1000.0
    return value


def _parse_retry_after(headers: Message | None) -> float | None:
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _retry_delay_s(attempt: int, detail: str, headers: Message | None) -> float:
    """Pista do servidor quando existe; senão backoff exponencial.

    Se o corpo e o ``Retry-After`` discordam, espera o maior: os dois
    são a API dizendo quanto falta, e o menor corta a janela cedo.
    """
    candidates = [
        value
        for value in (
            _parse_try_again_seconds(detail),
            _parse_retry_after(headers),
        )
        if value is not None
    ]
    if candidates:
        return _bounded_delay(max(candidates))
    return _bounded_delay(_RETRY_BASE_S * (2**attempt))


def _retryable_status(code: int) -> bool:
    # 403 da Cloudflare (1010) não melhora com repetição. Quem resolve é o User-Agent.
    return code == 429 or 500 <= code <= 599


def _transient_urlerror(exc: urllib.error.URLError) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return False
    reason = exc.reason
    # SSLError é OSError, mas certificado inválido não passa com repetição.
    if isinstance(reason, ssl.SSLError):
        return False
    if isinstance(reason, (OSError, http.client.IncompleteRead)):
        return True
    text = str(reason).lower()
    return any(marker in text for marker in _TRANSIENT_REASON_MARKERS)


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read()
    except Exception:
        return ""
    finally:
        try:
            exc.close()
        except Exception:
            pass
    if isinstance(payload, str):
        return payload
    return payload.decode("utf-8", errors="replace")


def _give_up(message: str, *, exhausted: bool, attempts: int) -> str:
    if exhausted:
        return f"{message} (após {attempts} tentativas)"
    return message


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
    encoded = json.dumps(
        {
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
    ).encode("utf-8")
    url = f"{base}/chat/completions"
    raw: bytes | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": LLM_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            break
        except urllib.error.HTTPError as exc:
            detail = _error_body(exc)
            exhausted = attempt + 1 >= LLM_MAX_ATTEMPTS
            if exhausted or not _retryable_status(exc.code):
                raise RuntimeError(
                    _give_up(
                        f"LLM HTTP {exc.code}: {detail[:500]}",
                        exhausted=exhausted and _retryable_status(exc.code),
                        attempts=attempt + 1,
                    )
                ) from exc
            time.sleep(_retry_delay_s(attempt, detail, exc.headers))
        except urllib.error.URLError as exc:
            exhausted = attempt + 1 >= LLM_MAX_ATTEMPTS
            transient = _transient_urlerror(exc)
            if exhausted or not transient:
                raise RuntimeError(
                    _give_up(
                        f"Falha ao chamar o LLM: {exc.reason}",
                        exhausted=exhausted and transient,
                        attempts=attempt + 1,
                    )
                ) from exc
            time.sleep(_retry_delay_s(attempt, "", None))
        except (TimeoutError, ConnectionError, http.client.IncompleteRead) as exc:
            exhausted = attempt + 1 >= LLM_MAX_ATTEMPTS
            if exhausted:
                raise RuntimeError(
                    _give_up(
                        f"Falha ao chamar o LLM: {exc}",
                        exhausted=True,
                        attempts=attempt + 1,
                    )
                ) from exc
            time.sleep(_retry_delay_s(attempt, "", None))
    if raw is None:
        raise RuntimeError("Falha ao chamar o LLM.")
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
