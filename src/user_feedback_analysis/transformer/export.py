"""Empacota o checkpoint do Transformer no formato do Hugging Face Hub.

O encoder e o tokenizer vão no layout padrão do Hub. A cabeça linear fica
em ``classifier_head.safetensors`` (o modelo não é AutoModelForSequenceClassification).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from user_feedback_analysis.transformer.dataset import ID_TO_LABEL, LABEL_TO_ID
from user_feedback_analysis.transformer.model import DEFAULT_ENCODER, SatisfactionModel

HF_REPO_ID = "pln-ufrgs/ufa-wildfeedback-satisfaction-mbert"
CLASSIFIER_FILENAME = "classifier_head.safetensors"
LABEL_MAP_FILENAME = "label_map.json"
RUN_SUMMARY_FILENAME = "run_summary.json"


def label_map() -> dict[str, str]:
    return {str(index): label.value for index, label in ID_TO_LABEL.items()}


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "O export do Transformer precisa de torch. Instale requirements.txt."
        ) from exc
    return torch


def load_checkpoint(
    checkpoint: Path | str,
    *,
    encoder_name: str = DEFAULT_ENCODER,
    config: object | None = None,
    tokenizer=None,
    local_files_only: bool = False,
) -> tuple[SatisfactionModel, object]:
    """Carrega ``best.pt`` (state_dict do SatisfactionModel) e o tokenizer."""
    torch = _require_torch()
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "O export precisa de transformers. Instale requirements.txt."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            encoder_name,
            local_files_only=local_files_only,
        )
    model = SatisfactionModel(
        encoder_name,
        config=config,
        local_files_only=local_files_only,
    )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def export_for_hub(
    checkpoint: Path | str,
    output_dir: Path | str,
    *,
    encoder_name: str = DEFAULT_ENCODER,
    config: object | None = None,
    tokenizer=None,
    run_record: dict[str, Any] | None = None,
    local_files_only: bool = False,
) -> Path:
    """Grava encoder, tokenizer, cabeça, label_map e run_summary em ``output_dir``."""
    torch = _require_torch()
    model, tokenizer = load_checkpoint(
        checkpoint,
        encoder_name=encoder_name,
        config=config,
        tokenizer=tokenizer,
        local_files_only=local_files_only,
    )
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    head = {key: value.detach().cpu().contiguous() for key, value in model.classifier.state_dict().items()}
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "O export precisa de safetensors. Instale requirements.txt."
        ) from exc
    save_file(head, dest / CLASSIFIER_FILENAME)
    (dest / LABEL_MAP_FILENAME).write_text(
        json.dumps(label_map(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = _run_summary(run_record, encoder_name=model.encoder_name)
    (dest / RUN_SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del torch
    return dest


def load_exported(
    folder: Path | str,
    *,
    encoder_name: str = "exported",
) -> tuple[SatisfactionModel, object]:
    """Recarrega encoder + cabeça a partir de uma pasta de export."""
    torch = _require_torch()
    path = Path(folder)
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "O carregamento precisa de transformers e safetensors."
        ) from exc
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = SatisfactionModel(encoder_name, config=config)
    model.encoder = AutoModel.from_pretrained(path)
    head = load_file(str(path / CLASSIFIER_FILENAME))
    model.classifier.load_state_dict(head)
    model.eval()
    del torch
    return model, tokenizer


def _run_summary(
    run_record: dict[str, Any] | None,
    *,
    encoder_name: str,
) -> dict[str, Any]:
    if not run_record:
        return {
            "encoder": encoder_name,
            "num_labels": 3,
            "label_map": label_map(),
            "note": "Sem JSON de corrida neste export.",
        }
    metrics = dict(run_record.get("metrics") or {})
    training = dict(run_record.get("training") or {})
    split = dict(run_record.get("split") or {})
    per_class = metrics.get("per_class") or {}
    f1_by_class = {
        name: float(scores["f1"])
        for name, scores in per_class.items()
        if isinstance(scores, dict) and "f1" in scores
    }
    return {
        "seed": split.get("seed"),
        "encoder": encoder_name,
        "best_epoch": training.get("best_epoch"),
        "best_val_macro_f1": training.get("best_val_macro_f1"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_macro_precision": metrics.get("macro_precision"),
        "test_macro_recall": metrics.get("macro_recall"),
        "test_f1_per_class": f1_by_class,
        "n_test": metrics.get("n"),
        "official_result": run_record.get("official_result"),
        "label_map": label_map(),
        "label_to_id": {label.value: index for label, index in LABEL_TO_ID.items()},
    }


def push_export_folder(
    folder: Path | str,
    *,
    repo_id: str = HF_REPO_ID,
    path_in_repo: str,
    commit_message: str | None = None,
) -> str:
    """Sobe a pasta exportada para ``path_in_repo`` no Hub. Usa o login local."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("O upload precisa de huggingface_hub.") from exc
    api = HfApi()
    message = commit_message or f"Add Transformer export at {path_in_repo}"
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=message,
    )
    return f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
