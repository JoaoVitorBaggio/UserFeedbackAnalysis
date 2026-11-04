"""Harness único das quatro corridas.

Entrada: rótulos ouro e rótulos preditos em {SAT, DSAT, sem sinal}.
Saída: precisão, recall e macro F1, por classe e na média macro.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from user_feedback_analysis.metrics import HarnessReport, evaluate, format_report

__all__ = ["HarnessReport", "evaluate", "format_report", "main", "read_label_file"]


def read_label_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} não é uma lista de rótulos.")
        return [str(item) for item in data]
    return [line.strip() for line in text.splitlines() if line.strip()]


def write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prediction_rows(
    example_ids: Sequence[str],
    y_true: Sequence[object],
    y_pred: Sequence[object],
) -> list[dict[str, str]]:
    if not (len(example_ids) == len(y_true) == len(y_pred)):
        raise ValueError("Ids, ouro e predição precisam do mesmo tamanho.")
    return [
        {"id": example_id, "gold": str(gold), "pred": str(pred)}
        for example_id, gold, pred in zip(example_ids, y_true, y_pred)
    ]


def build_run_record(
    *,
    arm: str,
    report: HarnessReport,
    split: dict[str, object],
    predictions: list[dict[str, str]],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Registro de uma corrida.

    ``official_result`` sai falso. O arquivo não é o registro do artigo.
    O runner de LLM pode sobrescrever o campo quando a chamada casa com
    a decisão de 2026-11-03. Cardiff e Transformer não fazem isso.
    """
    payload: dict[str, object] = {
        "arm": arm,
        "official_result": False,
        "comparison_metric": "macro_f1",
        "metrics": report.to_dict(),
        "predictions": predictions,
        "split": split,
    }
    if extra:
        payload.update(extra)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Harness compartilhado: precisão, recall e macro F1 "
            "em SAT, DSAT e sem sinal."
        )
    )
    parser.add_argument("--gold", type=Path, required=True, help="Um rótulo por linha, ou uma lista JSON.")
    parser.add_argument("--pred", type=Path, required=True, help="Rótulos preditos, na mesma ordem do ouro.")
    parser.add_argument("--output", type=Path, default=None, help="JSON opcional com o relatório.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        y_true = read_label_file(args.gold)
        y_pred = read_label_file(args.pred)
        report = evaluate(y_true, y_pred)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    if args.output is not None:
        write_json(
            build_run_record(
                arm="harness",
                report=report,
                split={},
                predictions=prediction_rows(
                    [str(i) for i in range(len(y_true))],
                    y_true,
                    y_pred,
                ),
            ),
            args.output,
        )
        print(f"relatório: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
