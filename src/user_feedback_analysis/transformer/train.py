"""Ponto de entrada do treino. O loop não está implementado de propósito.

Contrato para o Nikolas Chemale:

- Entrada: ``train`` e, se houver, ``val``, saídos de ``make_splits``.
- O teste oficial não entra no treino nem no early stopping.
- Loss: ``build_loss()``, cross-entropy categórica, 3 classes, sem ponderação.
- Cabeça: 3 logits, índices em ``LABEL_TO_ID``.
- Avaliação: ``user_feedback_analysis.harness.evaluate`` no hold-out, no fim.
- GPU de referência do João: GTX 1060 6GB. Encoder base, batch pequeno e
  sequência curta são o limite de hardware, não um hiperparâmetro medido.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from user_feedback_analysis.records import Example


def train_transformer(
    train: Sequence[Example],
    val: Sequence[Example] | None = None,
    *,
    encoder_name: str = "a definir",
    output_dir: Path | str | None = None,
    seed: int | None = None,
) -> None:
    del train, val, encoder_name, output_dir, seed
    raise NotImplementedError(
        "O treino do Transformer é do Nikolas Chemale (braço núcleo). "
        "Use make_splits, SatisfactionDataset, build_loss e o harness. "
        "Não leia o teste no treino nem no early stopping."
    )
