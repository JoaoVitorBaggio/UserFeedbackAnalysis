"""Corridas de LLM zero-shot e few-shot. A métrica sai do harness compartilhado.

Sem ``OPENAI_API_KEY`` a corrida entra em dry-run e não chama a rede.
A chamada real exige ``LLM_MODEL``. O id fechado em 2026-11-03 é
``openai/gpt-oss-20b``. Desde 2026-11-04 a base aceita é a DeepInfra
(``https://api.deepinfra.com/v1/openai``); a base da Groq foi revogada.
Llama 3.1 8B Instruct já tinha sido revogado.
Cada exemplo avaliado recebe K inferências (padrão 3); o rótulo é a
moda. ``--k`` do few-shot é outra coisa: quantas demonstrações entram
no prompt. O padrão fechado é 3; antes estava em aberto.

``--test-subsample N`` tira N turnos do teste oficial, estratificados,
depois do split. Sem a flag, o teste oficial inteiro é avaliado. A
flag não muda o treino.

Recusa ou ``content`` vazio, esgotadas as tentativas daquela
inferência, conta como sem sinal (2026-11-06). A semente segue. O
caso entra na predição, no checkpoint e em ``protocol_deviations``.
``official_result`` fica falso. HTTP 429, HTTP 5xx e falha de rede
continuam parando a semente; o checkpoint do prefixo fica no disco.

Com ``--output``, cada exemplo concluído substitui esse arquivo por
um checkpoint (replace atômico). A mesma invocação, no mesmo caminho,
continua do ``next_index`` e não refaz o voto já gravado. Sem
``--output`` não há o que retomar. O checkpoint não é o JSON final:
``official_result`` nele fica falso. A regra do resultado oficial não
olha para a retomada.

    python -m user_feedback_analysis.llm.run \\
        --data tests/fixtures/tiny_sat_dsat.jsonl \\
        --shot zero --test-size 3 --per-class 1 --seed 0 --dry-run

    python -m user_feedback_analysis.llm.run \\
        --data tests/fixtures/tiny_sat_dsat.jsonl \\
        --shot few --k 3 --test-size 3 --per-class 1 --seed 0 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from user_feedback_analysis.cli_args import (
    add_corpus_args,
    add_split_args,
    config_from_args,
    load_corpus,
)
from user_feedback_analysis.data.split import format_split, make_splits
from user_feedback_analysis.data.subsample import (
    absent_subsample_record,
    format_subsample,
    subsample_stratified,
)
from user_feedback_analysis.harness import (
    build_run_record,
    evaluate,
    format_report,
    prediction_rows,
    write_json,
)
from user_feedback_analysis.llm.checkpoint import (
    build_fingerprint,
    load_resume,
    write_checkpoint,
)
from user_feedback_analysis.llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_INFERENCE_K,
    DEFAULT_TEMPERATURE,
    VoteResult,
    classify_votes,
    resolve_dry_run,
)
from user_feedback_analysis.llm.protocol import (
    OFFICIAL_RESULT_POLICY,
    PROTOCOL_FEW_SHOT_K,
    PROTOCOL_LLM_BASE_URL,
    PROTOCOL_LLM_MODEL_ID,
    apply_refusal_deviation,
    llm_official_result,
    model_requirement_record,
    public_base_url,
)
from user_feedback_analysis.llm.prompts import clip_text
from user_feedback_analysis.llm.sample import sample_demonstrations
from user_feedback_analysis.llm.vote import TIE_BREAK, VOTE
from user_feedback_analysis.records import Example


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM zero-shot ou few-shot no espaço SAT, DSAT e sem sinal."
    )
    add_corpus_args(parser)
    add_split_args(parser)
    parser.add_argument("--shot", choices=("zero", "few"), required=True)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Demonstrações do few-shot, só do treino. "
            f"O padrão é {PROTOCOL_FEW_SHOT_K}, fechado em 2026-11-03 "
            "(antes estava em aberto). Não é o K=3 das inferências. "
            "Zero-shot recusa esta flag."
        ),
    )
    parser.add_argument(
        "--test-subsample",
        type=int,
        default=None,
        help=(
            "Depois do split, avalia só N turnos do teste oficial, "
            "estratificados pela mistura desse teste (maiores restos). "
            "Não muda o treino nem o hold-out dos outros braços. "
            "A decisão de 2026-11-03 usa 1000. Sem a flag, o teste inteiro entra."
        ),
    )
    parser.add_argument(
        "--inferences",
        type=int,
        default=DEFAULT_INFERENCE_K,
        help=(
            "Inferências por exemplo avaliado. O protocolo usa K=3, agregadas "
            "por voto majoritário. Não é o --k do few-shot."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Temperatura de cada inferência. O padrão do cliente é 0.7, "
            "para as K amostras poderem divergir; não é número do protocolo. "
            "0 continua aceito. O valor entra no JSON."
        ),
    )
    parser.add_argument(
        "--few-shot-seed",
        type=int,
        default=None,
        help="Semente da amostra de exemplos. Padrão: a mesma --seed do split.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Não chama a rede.")
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Exige OPENAI_API_KEY e LLM_MODEL. Na DeepInfra, "
            f"OPENAI_BASE_URL={PROTOCOL_LLM_BASE_URL} e "
            f"LLM_MODEL={PROTOCOL_LLM_MODEL_ID}. A chave não entra no código."
        ),
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Corte de cada texto no prompt. 0 desliga. Não é decisão do protocolo.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload, report_text, split_text = run_llm(args)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(split_text)
    print(report_text)
    if payload["dry_run"]:
        print("aviso: dry-run. As predições são mock e não são resultado experimental.")
    elif not payload["official_result"]:
        print(
            "aviso: official_result false. Os desvios estão em protocol_deviations."
        )
    if args.output is not None:
        try:
            write_json(payload, args.output)
        except OSError as exc:
            print(f"erro: {exc}", file=sys.stderr)
            print(
                f"checkpoint preservado em {args.output}. Rode de novo para concluir.",
                file=sys.stderr,
            )
            return 1
        print(f"relatório: {args.output}")
    return 0


def run_llm(args: argparse.Namespace) -> tuple[dict[str, object], str, str]:
    if args.dry_run and args.live:
        raise ValueError("Use só um entre --dry-run e --live.")
    if args.shot == "zero" and args.k is not None:
        raise ValueError("Zero-shot não usa --k.")
    if args.shot == "few":
        few_shot_k = PROTOCOL_FEW_SHOT_K if args.k is None else args.k
    else:
        few_shot_k = 0
    if isinstance(args.inferences, bool) or not isinstance(args.inferences, int) or args.inferences < 1:
        raise ValueError("inferences é um inteiro >= 1. O protocolo usa K=3.")
    if (
        isinstance(args.temperature, bool)
        or not isinstance(args.temperature, (int, float))
        or float(args.temperature) < 0
    ):
        raise ValueError("temperature é um número >= 0. O protocolo não exige 0.")
    if args.max_chars < 0:
        raise ValueError("max_chars é >= 0. Zero desliga o corte.")
    if args.test_subsample is not None and (
        isinstance(args.test_subsample, bool)
        or not isinstance(args.test_subsample, int)
        or args.test_subsample < 1
    ):
        raise ValueError("test_subsample é um inteiro >= 1.")
    max_chars = None if args.max_chars == 0 else args.max_chars
    requested = True if args.dry_run else False if args.live else None
    dry_run = resolve_dry_run(requested)

    corpus = load_corpus(args)
    config = config_from_args(args)
    split = make_splits(corpus, config)
    few_seed = args.seed if args.few_shot_seed is None else args.few_shot_seed
    demonstrations: list[Example] = []
    if args.shot == "few":
        demonstrations = sample_demonstrations(split.train, few_shot_k, few_seed)
        test_ids = {example.example_id for example in split.test}
        leaked = [example.example_id for example in demonstrations if example.example_id in test_ids]
        if leaked:
            raise RuntimeError(f"Few-shot viu o teste: {leaked}.")

    subsample_result = None
    evaluated = split.test
    if args.test_subsample is not None:
        subsample_result = subsample_stratified(split.test, args.test_subsample, args.seed)
        evaluated = subsample_result.examples

    model_id = os.environ.get("LLM_MODEL")
    if isinstance(model_id, str):
        model_id = model_id.strip() or None
    raw_base = os.environ.get("OPENAI_BASE_URL")
    base_url = public_base_url(raw_base, fallback=DEFAULT_BASE_URL)
    official, deviations = llm_official_result(
        dry_run=dry_run,
        model_id=model_id,
        base_url=raw_base,
        shot=args.shot,
        few_shot_k=few_shot_k,
        inferences=args.inferences,
        test_subsample=args.test_subsample,
        test_fraction=config.test_fraction,
        per_class=config.per_class,
        strict_per_class=config.strict_per_class,
        seed=config.seed,
        fallback_base_url=DEFAULT_BASE_URL,
    )

    fingerprint = build_fingerprint(
        shot=args.shot,
        seed=config.seed,
        few_shot_seed=few_seed if args.shot == "few" else None,
        few_shot_k=few_shot_k,
        inferences=args.inferences,
        temperature=float(args.temperature),
        max_chars=max_chars,
        dry_run=dry_run,
        model_id=model_id,
        base_url=base_url,
        test_subsample=args.test_subsample,
        test_fraction=config.test_fraction,
        test_size=config.test_size,
        per_class=config.per_class,
        strict_per_class=config.strict_per_class,
        split_unit=config.split_unit,
        role=config.role,
        val_size=config.val_size,
        demonstrations=demonstrations,
        evaluated=evaluated,
    )
    resumed = (
        load_resume(args.output, fingerprint, evaluated)
        if args.output is not None
        else ()
    )
    n_reused = len(resumed)
    if n_reused:
        print(
            f"retomada: {n_reused} de {len(evaluated)} exemplos em {args.output}",
            flush=True,
        )
    voted = list(resumed)
    for example in evaluated[n_reused:]:
        result = classify_votes(
            example.text,
            demonstrations,
            dry_run=dry_run,
            model=model_id,
            max_chars=max_chars,
            timeout=args.timeout,
            k=args.inferences,
            temperature=float(args.temperature),
        )
        voted.append(result)
        if args.output is not None:
            write_checkpoint(args.output, fingerprint, evaluated, voted)
        print(
            f"exemplo {len(voted)}/{len(evaluated)} {example.example_id}",
            flush=True,
        )
    predictions = [result.label for result in voted]
    gold = [example.label for example in evaluated]
    report = evaluate(gold, predictions)
    refusal_cases = _refusal_cases(evaluated, voted)
    config_official = official
    official, deviations = apply_refusal_deviation(official, deviations, refusal_cases)
    n_test_truncated = _n_truncated([example.text for example in evaluated], max_chars)
    n_demo_truncated = _n_truncated([example.text for example in demonstrations], max_chars)
    if dry_run:
        note = (
            "Dry-run: predições mock, sem chamada de rede. "
            "official_result fica falso."
        )
    elif official:
        note = (
            "Chamada real no protocolo de LLM emendado em 2026-11-04 "
            "(openai/gpt-oss-20b na DeepInfra, subamostra estratificada de 1000). "
            "official_result true: a chamada casa com esse protocolo. "
            "O JSON continua local e não é sozinho o registro do artigo."
        )
    elif config_official:
        note = (
            "Chamada nos botões do protocolo de LLM emendado em 2026-11-04 "
            "(openai/gpt-oss-20b na DeepInfra, subamostra estratificada de 1000). "
            "Recusa persistente foi contada como sem sinal. "
            "official_result false: o desvio fica visível e a corrida entra no relatório. "
            "Os desvios estão em protocol_deviations."
        )
    else:
        note = (
            "Chamada fora do protocolo de LLM emendado em 2026-11-04. "
            "official_result false. Os desvios estão em protocol_deviations."
        )
        if refusal_cases:
            note += (
                " Recusa persistente também foi contada como sem sinal "
                "e está em protocol_deviations."
            )
    if n_reused:
        note += (
            f" Retomada: {n_reused} exemplos já estavam no --output "
            "e não foram inferidos de novo."
        )
    if subsample_result is None:
        test_subsample = absent_subsample_record(
            split.manifest.test_class_counts,
            split.manifest.n_test,
        )
    else:
        test_subsample = subsample_result.to_dict()
    payload = build_run_record(
        arm="llm_zero_shot" if args.shot == "zero" else "llm_few_shot",
        report=report,
        split=split.manifest.to_dict(),
        predictions=_prediction_rows(
            [example.example_id for example in evaluated],
            gold,
            predictions,
            voted,
        ),
        extra={
            "dry_run": dry_run,
            "live": not dry_run,
            "model": "mock" if dry_run else model_id,
            "model_id": model_id,
            "base_url": base_url,
            "model_requirement": model_requirement_record(),
            "official_result": official,
            "official_result_policy": OFFICIAL_RESULT_POLICY,
            "protocol_deviations": deviations,
            "shot": args.shot,
            "k": few_shot_k,
            "few_shot_seed": few_seed if args.shot == "few" else None,
            "demonstration_ids": [example.example_id for example in demonstrations],
            "max_chars": max_chars,
            "n_test_truncated": n_test_truncated,
            "n_demo_truncated": n_demo_truncated,
            "inference_k": args.inferences,
            "vote": VOTE,
            "tie_break": TIE_BREAK,
            "temperature": float(args.temperature),
            "n_tie_breaks": sum(1 for result in voted if result.decision == TIE_BREAK),
            "inference_votes": [
                [label.value for label in result.votes] for result in voted
            ],
            "test_subsample": test_subsample,
            "run_fingerprint": fingerprint,
            "resume": {"n_reused": n_reused},
            "note": note,
        },
    )
    split_text = format_split(split.manifest)
    if subsample_result is not None:
        split_text = split_text + "\n" + format_subsample(subsample_result)
    return payload, format_report(report), split_text


def _n_truncated(texts: list[str], max_chars: int | None) -> int:
    return sum(1 for text in texts if clip_text(text, max_chars)[1])


def _refusal_cases(
    examples: Sequence[Example],
    voted: Sequence[VoteResult],
) -> list[tuple[str, tuple[int, ...]]]:
    cases: list[tuple[str, tuple[int, ...]]] = []
    for example, result in zip(examples, voted):
        if not result.refusals:
            continue
        cases.append((example.example_id, tuple(item.index for item in result.refusals)))
    return cases


def _prediction_rows(
    example_ids: Sequence[str],
    y_true: Sequence[object],
    y_pred: Sequence[object],
    voted: Sequence[VoteResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        dict(row)
        for row in prediction_rows(example_ids, y_true, y_pred)
    ]
    if len(rows) != len(voted):
        raise RuntimeError("Predições e votos divergem.")
    for row, result in zip(rows, voted):
        if not result.refusals:
            continue
        row["refusal_fallback"] = True
        row["refusal_responses"] = [
            {"index": item.index, "text": item.raw_response} for item in result.refusals
        ]
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
