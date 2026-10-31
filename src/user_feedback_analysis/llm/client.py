"""Cliente compatível com a API de chat da OpenAI.

Variáveis de ambiente:

- ``OPENAI_API_KEY``: obrigatória na chamada real.
- ``LLM_MODEL``: obrigatória na chamada real. O protocolo deixa o modelo a definir.
- ``OPENAI_BASE_URL``: opcional. Padrão ``https://api.openai.com/v1``.
  Serve endpoint local (por exemplo um servidor compatível).

Sem chave, a corrida cai no dry-run e não chama a rede.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from user_feedback_analysis.labels import Label
from user_feedback_analysis.llm.mock import mock_label
from user_feedback_analysis.llm.parse import parse_label
from user_feedback_analysis.records import Example

DEFAULT_BASE_URL = "https://api.openai.com/v1"


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
            raise RuntimeError(
                "LLM_MODEL ausente. O protocolo deixa o modelo a definir; "
                "passe o id pela variável LLM_MODEL."
            )
        return False
    return True


def chat_complete(prompt: str, *, model: str, timeout: float = 60.0) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY ausente.")
    base = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    body = {
        "model": model,
        "temperature": 0,
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


def classify_text(
    text: str,
    demonstrations: list[Example],
    *,
    dry_run: bool,
    model: str | None,
    max_chars: int | None,
    timeout: float = 60.0,
) -> Label:
    from user_feedback_analysis.llm.prompts import build_prompt, clip_text

    prompt, _ = build_prompt(text, demonstrations, max_chars=max_chars)
    if dry_run:
        shown, _ = clip_text(text, max_chars)
        return mock_label(shown)
    if not model:
        raise RuntimeError("LLM_MODEL ausente.")
    raw = chat_complete(prompt, model=model, timeout=timeout)
    return parse_label(raw)
