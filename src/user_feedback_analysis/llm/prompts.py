"""Prompts em inglês: o card declara o corpus em inglês.

O texto classificado é o turno do usuário (feedback sobre a resposta do
assistente), que é a pergunta do trabalho. O braço não recebe gradiente.
"""

from __future__ import annotations

from typing import Sequence

from user_feedback_analysis.labels import Label
from user_feedback_analysis.records import Example

_INSTRUCTIONS = """\
Classify the user message into exactly one label: SAT, DSAT, or sem sinal.

The message is the user's turn in a chat with an AI assistant.
SAT: the user shows satisfaction with the assistant's reply.
DSAT: the user shows dissatisfaction, correction, or rejection of the assistant's reply.
sem sinal: the message has no clear satisfaction or dissatisfaction signal.

Reply with only one label and no other text.\
"""


def clip_text(text: str, max_chars: int | None) -> tuple[str, bool]:
    if max_chars is None:
        return text, False
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars é um inteiro >= 1, ou fica omitido.")
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _fence(text: str) -> str:
    return text.replace("</user_message>", "< /user_message>")


def build_prompt(
    text: str,
    demonstrations: Sequence[Example] = (),
    *,
    max_chars: int | None = None,
) -> tuple[str, int]:
    """Devolve o prompt e quantos textos foram cortados (consulta + exemplos)."""
    clipped, n_clipped = clip_text(text, max_chars)
    n_truncated = int(n_clipped)
    lines = [_INSTRUCTIONS]
    for index, example in enumerate(demonstrations, start=1):
        demo_text, demo_clipped = clip_text(example.text, max_chars)
        n_truncated += int(demo_clipped)
        lines.extend(
            [
                "",
                f"Example {index}",
                "User message:",
                _fence(demo_text),
                f"Label: {example.label.value}",
            ]
        )
    lines.extend(
        [
            "",
            "User message:",
            _fence(clipped),
            "Label:",
        ]
    )
    return "\n".join(lines), n_truncated
