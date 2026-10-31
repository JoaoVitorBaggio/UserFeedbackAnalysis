"""Precisão, recall e macro F1 no espaço fixo de três classes.

O macro F1 é a média aritmética dos F1 por classe, com as três classes sempre
presentes. Classe sem suporte entra com 0 (o mesmo critério de
``zero_division=0`` do scikit-learn quando ``labels`` é a lista fechada).
Acurácia não entra: o teste é desbalanceado e o protocolo não a usa para comparar.

A conta coincide com ``sklearn.metrics.f1_score(..., average="macro", zero_division=0)``
quando ``labels`` é a lista fechada das três classes. Isso checa a fórmula; não é
métrica do corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from user_feedback_analysis.labels import LABELS, Label, normalize_label


@dataclass(frozen=True)
class ClassScore:
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class HarnessReport:
    label_names: tuple[str, ...]
    per_class: dict[str, ClassScore]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    n: int
    confusion: tuple[tuple[int, ...], ...]
    zero_division: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.label_names),
            "per_class": {name: score.to_dict() for name, score in self.per_class.items()},
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "n": self.n,
            "confusion": [list(row) for row in self.confusion],
            "confusion_axes": {"rows": "gold", "columns": "predicted"},
            "zero_division": self.zero_division,
            "comparison_metric": "macro_f1",
        }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    if total == 0.0:
        return 0.0
    return 2.0 * precision * recall / total


def evaluate(
    y_true: Sequence[Label | str],
    y_pred: Sequence[Label | str],
    *,
    labels: Sequence[Label] = LABELS,
) -> HarnessReport:
    """Avalia predições contra o ouro. As quatro corridas entram por aqui."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Ouro e predição têm tamanhos diferentes: {len(y_true)} e {len(y_pred)}."
        )
    label_list = [normalize_label(label) for label in labels]
    if not label_list:
        raise ValueError("O harness precisa da lista de classes.")
    if len(set(label_list)) != len(label_list):
        raise ValueError("A lista de classes do harness tem rótulo repetido.")

    gold = [normalize_label(value) for value in y_true]
    pred = [normalize_label(value) for value in y_pred]
    index = {label: pos for pos, label in enumerate(label_list)}
    unknown = [value for value in gold + pred if value not in index]
    if unknown:
        raise ValueError(
            "Há rótulo fora do espaço SAT, DSAT, sem sinal: "
            + ", ".join(sorted({item.value for item in unknown}))
        )

    size = len(label_list)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for true_label, pred_label in zip(gold, pred):
        matrix[index[true_label]][index[pred_label]] += 1

    per_class: dict[str, ClassScore] = {}
    for pos, label in enumerate(label_list):
        tp = matrix[pos][pos]
        predicted = sum(matrix[row][pos] for row in range(size))
        support = sum(matrix[pos])
        fp = predicted - tp
        fn = support - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        per_class[label.value] = ClassScore(
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            support=support,
            predicted=predicted,
            tp=tp,
            fp=fp,
            fn=fn,
        )

    n_labels = len(label_list)
    return HarnessReport(
        label_names=tuple(label.value for label in label_list),
        per_class=per_class,
        macro_precision=sum(score.precision for score in per_class.values()) / n_labels,
        macro_recall=sum(score.recall for score in per_class.values()) / n_labels,
        macro_f1=sum(score.f1 for score in per_class.values()) / n_labels,
        n=len(gold),
        confusion=tuple(tuple(row) for row in matrix),
    )


def format_report(report: HarnessReport) -> str:
    header = f"{'classe':<12} {'precisão':>10} {'recall':>10} {'f1':>10} {'suporte':>8}"
    lines = [header]
    for name in report.label_names:
        score = report.per_class[name]
        lines.append(
            f"{name:<12} {score.precision:10.4f} {score.recall:10.4f} "
            f"{score.f1:10.4f} {score.support:8d}"
        )
    lines.append(
        f"{'macro':<12} {report.macro_precision:10.4f} {report.macro_recall:10.4f} "
        f"{report.macro_f1:10.4f} {report.n:8d}"
    )
    lines.append("Métrica de comparação: macro F1.")
    return "\n".join(lines)
