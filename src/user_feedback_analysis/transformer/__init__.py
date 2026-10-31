"""Interfaces do braço núcleo. O treino é do Nikolas Chemale.

O que já está aqui para ele reutilizar: partição (``make_splits``), dataset de
exemplos com índice de classe, e ``build_loss`` (cross-entropy categórica, 3
classes, sem ponderação). ``train_transformer`` e o forward do modelo levantam
``NotImplementedError``.
"""

from user_feedback_analysis.transformer.dataset import LABEL_TO_ID, ID_TO_LABEL, SatisfactionDataset
from user_feedback_analysis.transformer.loss import LOSS_NAME, N_CLASSES, build_loss
from user_feedback_analysis.transformer.model import SatisfactionModel
from user_feedback_analysis.transformer.train import train_transformer

__all__ = [
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "LOSS_NAME",
    "N_CLASSES",
    "SatisfactionDataset",
    "SatisfactionModel",
    "build_loss",
    "train_transformer",
]
