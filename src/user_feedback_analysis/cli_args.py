"""Argumentos compartilhados pelas corridas do João."""

from __future__ import annotations

import argparse
from pathlib import Path

from user_feedback_analysis.data.load import load_jsonl, load_wildfeedback
from user_feedback_analysis.data.split import PROTOCOL_TEST_FRACTION, SplitConfig
from user_feedback_analysis.records import LoadedCorpus


def add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        help="JSONL local com as colunas do sat_dsat_annotation.",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Carrega microsoft/WildFeedback, config sat_dsat_annotation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Prefixo de linhas. Smoke apenas; não é a distribuição real.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Revisão do dataset no Hub. Continua em aberto no protocolo.",
    )
    parser.add_argument("--cache-dir", default=None)


def add_split_args(parser: argparse.ArgumentParser) -> None:
    holdout = parser.add_mutually_exclusive_group()
    holdout.add_argument(
        "--test-size",
        type=int,
        default=None,
        help=(
            "Contagem inteira de exemplos no teste, sem estratificar. "
            "Serve para smoke e fixtures. Exclusivo com --test-fraction."
        ),
    )
    holdout.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help=(
            "Fração do teste em (0, 1), estratificada pelas três classes. "
            f"O protocolo usa {PROTOCOL_TEST_FRACTION}. O teste não é rebalanceado. "
            "Exclusivo com --test-size. Se nenhum dos dois vier, vale "
            f"{PROTOCOL_TEST_FRACTION}."
        ),
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10_000,
        help="Alvo do treino balanceado por classe. O protocolo mira cerca de 10 mil.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Semente desta invocação. O protocolo usa três sementes, as mesmas "
            "nas quatro corridas; cada corrida recebe uma. O inteiro entra no manifesto. "
            "Este CLI não escolhe os três inteiros."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help=(
            "Validação tirada do lado de treino, depois do teste, em contagem inteira. "
            "O tamanho continua em aberto no protocolo."
        ),
    )
    parser.add_argument(
        "--split-unit",
        choices=("conversation", "turn"),
        default="conversation",
        help="Padrão: conversa inferida. 'turn' pode vazar conversa entre teste e treino.",
    )
    parser.add_argument(
        "--role",
        default="User",
        help="User, Agent ou any. Padrão User é hipótese; o filtro segue em aberto.",
    )
    parser.add_argument(
        "--strict-per-class",
        action="store_true",
        help="Falha se alguma classe tiver menos que --per-class depois do hold-out.",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON da corrida. Não commitar.")


def load_corpus(args: argparse.Namespace) -> LoadedCorpus:
    if args.data and args.hf:
        raise ValueError("Use só um entre --data e --hf.")
    if args.hf:
        return load_wildfeedback(
            limit=args.limit,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
    if args.data is None:
        raise ValueError("Informe --data com um JSONL ou --hf para o dataset do Hub.")
    return load_jsonl(args.data, limit=args.limit)


def config_from_args(args: argparse.Namespace) -> SplitConfig:
    role = None if str(args.role).casefold() == "any" else str(args.role)
    test_size = getattr(args, "test_size", None)
    test_fraction = getattr(args, "test_fraction", None)
    if test_size is None and test_fraction is None:
        test_fraction = PROTOCOL_TEST_FRACTION
    return SplitConfig(
        seed=args.seed,
        test_size=test_size,
        test_fraction=test_fraction,
        per_class=args.per_class,
        val_size=args.val_size,
        split_unit=args.split_unit,
        role=role,
        strict_per_class=args.strict_per_class,
    )
