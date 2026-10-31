"""Placeholder do encoder. Nikolas substitui pelo BERT/RoBERTa + cabeça linear.

A variante do encoder está a definir no protocolo. A cabeça tem 3 logits.
"""

from __future__ import annotations

from user_feedback_analysis.transformer.loss import N_CLASSES


class SatisfactionModel:
    def __init__(self, encoder_name: str = "a definir") -> None:
        if not encoder_name or not str(encoder_name).strip():
            raise ValueError("encoder_name vazio. O protocolo deixa a variante a definir.")
        self.encoder_name = encoder_name
        self.num_labels = N_CLASSES

    def forward(self, batch: object) -> object:
        raise NotImplementedError(
            "O forward do Transformer é do Nikolas Chemale. "
            f"Encoder pedido: {self.encoder_name}. "
            "A cabeça devolve 3 logits na ordem SAT, DSAT, sem sinal."
        )
