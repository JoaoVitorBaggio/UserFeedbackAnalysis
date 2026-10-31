"""Cache local de um prefixo ou do split inteiro. O arquivo fica fora do git."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from user_feedback_analysis.data.load import load_wildfeedback, write_jsonl


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Grava microsoft/WildFeedback (sat_dsat_annotation) em JSONL local. "
            "Não commitar o arquivo."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Prefixo de linhas, só para smoke. Omitir materializa o split inteiro.",
    )
    parser.add_argument("--revision", default=None, help="Revisão no Hub. O protocolo deixa a definir.")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        corpus = load_wildfeedback(
            limit=args.limit,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        write_jsonl(corpus, args.output)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(f"linhas: {len(corpus.rows)}")
    print(f"arquivo: {args.output}")
    if args.limit is not None:
        print("aviso: prefixo. Não é a distribuição real e não é o hold-out do experimento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
