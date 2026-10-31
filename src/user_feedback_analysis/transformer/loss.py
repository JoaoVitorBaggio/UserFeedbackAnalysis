"""Loss do Transformer fine-tunado.

Decisão do grupo, em docs/funcao-de-loss.md: cross-entropy categórica com 3
classes (SAT, DSAT, sem sinal). Sem sinal permanece classe própria. O caminho
mínimo não pondera: o treino já mira cerca de 10 mil exemplos por classe.

O baseline não treina neste corpus. O LLM não recebe gradiente aqui.
"""

from __future__ import annotations

from typing import Sequence

LOSS_NAME = "categorical_cross_entropy"
N_CLASSES = 3
USE_CLASS_WEIGHTS = False


def build_loss(*, class_weights: Sequence[float] | None = None):
    """Devolve ``torch.nn.CrossEntropyLoss`` sem peso de classe.

    Logits ``[batch, 3]``, alvo inteiro com ``LABEL_TO_ID``.
    """
    if class_weights is not None or USE_CLASS_WEIGHTS:
        raise ValueError(
            "O caminho mínimo não pondera classes. O balanceamento fica na amostra de treino."
        )
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "O critério de loss precisa de torch. Instale requirements.txt na máquina de treino."
        ) from exc
    criterion = nn.CrossEntropyLoss()
    if criterion.weight is not None:
        raise RuntimeError("CrossEntropyLoss do caminho mínimo sai sem weight.")
    return criterion
