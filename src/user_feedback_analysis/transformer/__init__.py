"""Braço núcleo: fine-tuning do Transformer em SAT, DSAT e sem sinal.

Reutiliza a partição (``make_splits``), o dataset com índice de classe e
``build_loss`` (cross-entropy categórica, 3 classes, sem ponderação).
O teste oficial não entra no treino nem no early stopping. A métrica da
corrida sai do harness compartilhado.
"""

from user_feedback_analysis.transformer.dataset import LABEL_TO_ID, ID_TO_LABEL, SatisfactionDataset
from user_feedback_analysis.transformer.loss import LOSS_NAME, N_CLASSES, build_loss
from user_feedback_analysis.transformer.model import DEFAULT_ENCODER, SatisfactionModel
from user_feedback_analysis.transformer.export import (
    HF_REPO_ID,
    export_for_hub,
    load_exported,
    push_export_folder,
)
from user_feedback_analysis.transformer.train import (
    HARDWARE_NOTE,
    TrainConfig,
    TrainResult,
    predict,
    set_seed,
    train_transformer,
)

__all__ = [
    "DEFAULT_ENCODER",
    "HF_REPO_ID",
    "HARDWARE_NOTE",
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "LOSS_NAME",
    "N_CLASSES",
    "SatisfactionDataset",
    "SatisfactionModel",
    "TrainConfig",
    "TrainResult",
    "build_loss",
    "export_for_hub",
    "load_exported",
    "predict",
    "push_export_folder",
    "set_seed",
    "train_transformer",
]
