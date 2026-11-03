"""Encoder BERT/E5 + cabeça linear de 3 logits (SAT, DSAT, sem sinal).

A cabeça lê o estado oculto do ``[CLS]``. O id do encoder entra no manifesto
da corrida. O padrão é BERT-base multilíngue; ``--encoder`` troca o id.
"""

from __future__ import annotations

from collections.abc import Mapping

from user_feedback_analysis.transformer.loss import N_CLASSES

DEFAULT_ENCODER = "bert-base-multilingual-cased"


def _nn_module_base():
    try:
        import torch.nn as nn
    except ImportError:
        return object
    return nn.Module


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "O encoder do Transformer precisa de torch. Instale requirements.txt na máquina de treino."
        ) from exc
    return torch, nn


def _require_auto_model():
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError(
            "O encoder do Transformer precisa de transformers. Instale requirements.txt na máquina de treino."
        ) from exc
    return AutoModel


class SatisfactionModel(_nn_module_base()):
    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        *,
        config: object | None = None,
        dropout: float | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not encoder_name or not str(encoder_name).strip():
            raise ValueError(
                "encoder_name vazio. Informe o id do encoder "
                f"(ex. {DEFAULT_ENCODER})."
            )
        _require_torch()
        AutoModel = _require_auto_model()
        super().__init__()
        self.encoder_name = str(encoder_name).strip()
        self.num_labels = N_CLASSES
        if config is not None:
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(
                self.encoder_name,
                local_files_only=local_files_only,
            )
        hidden = int(self.encoder.config.hidden_size)
        if dropout is None:
            dropout = float(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout é um número em [0, 1).")
        _, nn = _require_torch()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(hidden, N_CLASSES)

    def forward(self, batch: object) -> object:
        if not isinstance(batch, Mapping):
            raise TypeError("forward espera o dict do tokenizer (input_ids, attention_mask).")
        if "input_ids" not in batch:
            raise ValueError("O batch do Transformer precisa de input_ids.")
        kwargs: dict[str, object] = {"input_ids": batch["input_ids"]}
        if batch.get("attention_mask") is not None:
            kwargs["attention_mask"] = batch["attention_mask"]
        if batch.get("token_type_ids") is not None:
            kwargs["token_type_ids"] = batch["token_type_ids"]
        outputs = self.encoder(**kwargs)
        cls_state = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls_state))
