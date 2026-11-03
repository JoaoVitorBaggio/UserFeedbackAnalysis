"""Corrida do Transformer fine-tunado. A métrica sai do harness compartilhado.

    python -m user_feedback_analysis.transformer.run \\
        --data tests/fixtures/tiny_sat_dsat.jsonl \\
        --test-size 3 --per-class 1 --val-size 3 --seed 0 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

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
from user_feedback_analysis.transformer.model import DEFAULT_ENCODER
from user_feedback_analysis.transformer.train import (
    HARDWARE_NOTE,
    TrainConfig,
    predict,
    train_transformer,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tuning do Transformer em SAT, DSAT e sem sinal. "
            "O teste oficial não entra no treino nem no early stopping."
        )
    )
    add_corpus_args(parser)
    add_split_args(parser)
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help=(
            "Id do encoder no Hub. O protocolo aceita BERT-base multilíngue ou E5-base. "
            f"Padrão: {DEFAULT_ENCODER}."
        ),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Desliga AMP. O padrão é fp16 no CUDA (limite da RTX 4060 8GB).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Pasta do melhor state_dict. Sem este flag o melhor fica só na memória.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload, report_text, split_text = run_transformer(args)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(split_text)
    print(report_text)
    if args.output is not None:
        write_json(payload, args.output)
        print(f"relatório: {args.output}")
    return 0


def _require_val_size(args: argparse.Namespace) -> None:
    if getattr(args, "val_size", None) is None:
        raise ValueError(
            "O treino do Transformer exige --val-size. "
            "A validação sai do lado de treino, depois do teste. "
            "O teste oficial não entra no early stopping."
        )


def _assert_holdout_disjoint(split) -> None:
    test_ids = set(split.manifest.test_ids)
    train_ids = set(split.manifest.train_ids)
    val_ids = set(split.manifest.val_ids)
    leaked = test_ids & (train_ids | val_ids)
    if leaked:
        raise RuntimeError(
            "O teste oficial vazou para o treino ou a validação: "
            + ", ".join(sorted(leaked)[:8])
        )
    if not val_ids:
        raise ValueError(
            "A partição ficou sem validação. Aumente --val-size ou o corpus."
        )


def _encoder_info(model, encoder_name: str) -> dict[str, object]:
    config = model.encoder.config
    return {
        "id": encoder_name,
        "resolved_id": getattr(config, "_name_or_path", encoder_name),
        "revision": getattr(config, "_commit_hash", None),
        "hidden_size": int(config.hidden_size),
        "architectures": list(getattr(config, "architectures", None) or []),
        "head": "linear_on_cls",
        "num_labels": int(model.num_labels),
    }


def _runtime_info() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch_version": None, "cuda_available": False, "gpu_name": None}
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }


def run_transformer(args: argparse.Namespace) -> tuple[dict[str, object], str, str]:
    _require_val_size(args)
    encoder_name = str(args.encoder).strip()
    if not encoder_name:
        raise ValueError("encoder vazio.")
    corpus = load_corpus(args)
    config = config_from_args(args)
    split = make_splits(corpus, config)
    _assert_holdout_disjoint(split)

    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        fp16=not args.no_fp16,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    result = train_transformer(
        split.train,
        split.val,
        encoder_name=encoder_name,
        output_dir=args.checkpoint_dir,
        seed=args.seed,
        config=train_config,
    )
    predictions = predict(
        result.model,
        result.tokenizer,
        split.test,
        batch_size=train_config.batch_size,
        max_length=train_config.max_length,
        device=train_config.device,
    )
    gold = [example.label for example in split.test]
    report = evaluate(gold, predictions)
    payload = build_run_record(
        arm="transformer",
        report=report,
        split=split.manifest.to_dict(),
        predictions=prediction_rows(
            [example.example_id for example in split.test],
            gold,
            predictions,
        ),
        extra={
            "encoder": _encoder_info(result.model, result.encoder_name),
            "training": {
                "config": result.config.to_dict(),
                "best_epoch": result.best_epoch,
                "best_val_macro_f1": result.best_val_macro_f1,
                "history": result.history,
                "hardware_note": HARDWARE_NOTE,
                **_runtime_info(),
            },
        },
    )
    return payload, format_report(report), format_split(split.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
