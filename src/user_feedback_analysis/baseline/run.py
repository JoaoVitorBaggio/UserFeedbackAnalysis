"""Corrida do baseline. A métrica sai do harness compartilhado.

    python -m user_feedback_analysis.baseline.run \\
        --data tests/fixtures/tiny_sat_dsat.jsonl \\
        --test-size 3 --per-class 1 --seed 0 --mock
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from user_feedback_analysis.baseline.mapping import DEFAULT_MODEL_ID, mapping_dict
from user_feedback_analysis.baseline.mock import LexiconMockClassifier
from user_feedback_analysis.cli_args import (
    add_corpus_args,
    add_split_args,
    config_from_args,
    load_corpus,
)
from user_feedback_analysis.data.split import format_split, make_splits
from user_feedback_analysis.harness import (
    build_run_record,
    evaluate,
    format_report,
    prediction_rows,
    write_json,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline de sentimento pré-treinado, projetado em SAT, DSAT e sem sinal."
        )
    )
    add_corpus_args(parser)
    add_split_args(parser)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Léxico local, sem baixar o checkpoint. Não é resultado experimental.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload, report_text, split_text = run_baseline(args)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(split_text)
    print(report_text)
    if args.mock:
        print("aviso: classificador mock. Não é o baseline pré-treinado.")
    if args.output is not None:
        write_json(payload, args.output)
        print(f"relatório: {args.output}")
    return 0


def run_baseline(args: argparse.Namespace) -> tuple[dict[str, object], str, str]:
    corpus = load_corpus(args)
    config = config_from_args(args)
    split = make_splits(corpus, config)
    if args.mock:
        classifier = LexiconMockClassifier()
        predictions = classifier.predict([example.text for example in split.test])
        classifier_info: dict[str, object] = {
            "kind": classifier.kind,
            "model_id": None,
            "mapping": mapping_dict(),
            "note": "Léxico local para smoke. Não é o baseline pré-treinado.",
        }
    else:
        from user_feedback_analysis.baseline.classifier import (
            HuggingFaceSentimentClassifier,
        )

        classifier = HuggingFaceSentimentClassifier(
            args.model_id,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
        )
        predictions = classifier.predict([example.text for example in split.test])
        classifier_info = {
            "kind": classifier.kind,
            "model_id": classifier.model_id,
            "mapping": mapping_dict(),
            "preprocess": "cardiff-twitter" if classifier.preprocess else None,
            "max_length": classifier.max_length,
            "batch_size": classifier.batch_size,
            "device": classifier.device,
        }
    gold = [example.label for example in split.test]
    report = evaluate(gold, predictions)
    payload = build_run_record(
        arm="baseline",
        report=report,
        split=split.manifest.to_dict(),
        predictions=prediction_rows(
            [example.example_id for example in split.test],
            gold,
            predictions,
        ),
        extra={"classifier": classifier_info},
    )
    return payload, format_report(report), format_split(split.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
