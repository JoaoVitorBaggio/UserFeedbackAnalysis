"""Loop de fine-tuning do Transformer (braço núcleo).

Contrato:

- Entrada: ``train`` e ``val``, saídos de ``make_splits``. A validação é
  obrigatória: o early stopping usa macro F1 nela.
- O teste oficial não entra no treino nem no early stopping.
- Loss: ``build_loss()``, cross-entropy categórica, 3 classes, sem ponderação.
- Cabeça: 3 logits, índices em ``LABEL_TO_ID``.
- Avaliação de época: ``user_feedback_analysis.harness.evaluate`` na validação.
- GPU de referência: RTX 4060 8GB. Batch, max_length e fp16 são o limite de
  hardware, não um hiperparâmetro medido.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from user_feedback_analysis.harness import evaluate
from user_feedback_analysis.labels import Label
from user_feedback_analysis.records import Example
from user_feedback_analysis.transformer.dataset import (
    ID_TO_LABEL,
    SatisfactionDataset,
)
from user_feedback_analysis.transformer.loss import build_loss
from user_feedback_analysis.transformer.model import DEFAULT_ENCODER, SatisfactionModel

HARDWARE_NOTE = (
    "batch_size, max_length e fp16 são o limite da RTX 4060 8GB, "
    "não hiperparâmetros otimizados neste trabalho."
)


@dataclass
class TrainConfig:
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 1
    batch_size: int = 16
    grad_accum: int = 1
    max_length: int = 128
    fp16: bool = True
    device: str = "cpu"
    dropout: float | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        _require_count("epochs", self.epochs, minimum=1)
        _require_count("patience", self.patience, minimum=0)
        _require_count("batch_size", self.batch_size, minimum=1)
        _require_count("grad_accum", self.grad_accum, minimum=1)
        _require_count("max_length", self.max_length, minimum=1)
        if (
            isinstance(self.lr, bool)
            or not isinstance(self.lr, (int, float))
            or float(self.lr) <= 0
        ):
            raise ValueError("lr é um número > 0.")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or float(self.weight_decay) < 0
        ):
            raise ValueError("weight_decay é um número >= 0.")
        if (
            isinstance(self.warmup_ratio, bool)
            or not isinstance(self.warmup_ratio, (int, float))
            or not 0 <= float(self.warmup_ratio) < 1
        ):
            raise ValueError("warmup_ratio é um número em [0, 1).")
        device = str(self.device).strip().casefold()
        if device not in {"cpu", "cuda"}:
            raise ValueError("device é 'cpu' ou 'cuda'.")
        self.device = device

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TrainResult:
    model: SatisfactionModel
    tokenizer: object
    best_epoch: int
    history: list[dict[str, object]] = field(default_factory=list)
    encoder_name: str = DEFAULT_ENCODER
    config: TrainConfig = field(default_factory=TrainConfig)

    @property
    def best_val_macro_f1(self) -> float:
        if not self.history:
            return 0.0
        for row in self.history:
            if int(row["epoch"]) == self.best_epoch:
                return float(row["val_macro_f1"])
        return float(self.history[-1]["val_macro_f1"])


def _require_count(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} é um inteiro >= {minimum}.")


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "O treino do Transformer precisa de torch. Instale requirements.txt na máquina de treino."
        ) from exc
    return torch


def set_seed(seed: int) -> None:
    """Alinha random, numpy (se houver), torch e transformers na mesma semente."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed é um inteiro >= 0.")
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        from transformers import set_seed as huggingface_set_seed
    except ImportError:
        return
    huggingface_set_seed(seed)


def make_collate(tokenizer, max_length: int):
    """Tokeniza um lote de itens de ``SatisfactionDataset`` com padding dinâmico."""

    def collate(items: list[dict[str, object]]) -> dict[str, object]:
        torch = _require_torch()
        texts = [str(item["text"]) for item in items]
        labels = torch.tensor([int(item["label"]) for item in items], dtype=torch.long)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    return collate


def _move_batch(batch: dict[str, object], device: str) -> dict[str, object]:
    torch = _require_torch()
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _load_tokenizer(encoder_name: str, *, local_files_only: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "O treino do Transformer precisa de transformers. Instale requirements.txt na máquina de treino."
        ) from exc
    return AutoTokenizer.from_pretrained(encoder_name, local_files_only=local_files_only)


def _resolve_device(requested: str) -> str:
    torch = _require_torch()
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponível. Rode com --device cpu. "
            "Na RTX 4060 8GB, use --device cuda e batch que caiba na VRAM."
        )
    return "cuda"


def _param_groups(model: SatisfactionModel, weight_decay: float) -> list[dict[str, object]]:
    decay: list[object] = []
    no_decay: list[object] = []
    skip = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias", "layer_norm.bias")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in skip):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def predict(
    model: SatisfactionModel,
    tokenizer,
    examples: Sequence[Example],
    *,
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu",
) -> list[Label]:
    """Argmax dos logits → ``ID_TO_LABEL``. O teste só entra aqui, depois do treino."""
    if batch_size < 1:
        raise ValueError("batch_size é um inteiro >= 1.")
    if max_length < 1:
        raise ValueError("max_length é um inteiro >= 1.")
    if not examples:
        return []
    torch = _require_torch()
    device = _resolve_device(device)
    model.to(device)
    model.eval()
    collate = make_collate(tokenizer, max_length)
    labels: list[Label] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = collate(list(SatisfactionDataset(chunk)))
            batch.pop("labels")
            batch = _move_batch(batch, device)
            logits = model(batch)
            pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
            for index in pred_ids:
                if index not in ID_TO_LABEL:
                    raise RuntimeError(f"Logit argmax fora de LABEL_TO_ID: {index}.")
                labels.append(ID_TO_LABEL[index])
    if len(labels) != len(examples):
        raise RuntimeError("O modelo devolveu um número inesperado de rótulos.")
    return labels


def train_transformer(
    train: Sequence[Example],
    val: Sequence[Example] | None = None,
    *,
    encoder_name: str = DEFAULT_ENCODER,
    output_dir: Path | str | None = None,
    seed: int | None = None,
    config: TrainConfig | None = None,
    model: SatisfactionModel | None = None,
    tokenizer=None,
    model_config: object | None = None,
) -> TrainResult:
    if not train:
        raise ValueError("O treino do Transformer recebeu zero exemplos.")
    if val is None or len(val) == 0:
        raise ValueError(
            "O early stopping precisa de validação. "
            "Passe --val-size; o teste oficial não entra no treino nem no early stopping."
        )
    train_config = config if config is not None else TrainConfig()
    torch = _require_torch()
    device = _resolve_device(train_config.device)
    if seed is not None:
        set_seed(seed)

    if tokenizer is None:
        tokenizer = _load_tokenizer(
            encoder_name,
            local_files_only=train_config.local_files_only,
        )
    if model is None:
        model = SatisfactionModel(
            encoder_name,
            config=model_config,
            dropout=train_config.dropout,
            local_files_only=train_config.local_files_only,
        )
    model.to(device)

    from torch.utils.data import DataLoader

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        SatisfactionDataset(train),
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, train_config.max_length),
        generator=generator,
        num_workers=0,
    )
    criterion = build_loss()
    optimizer = torch.optim.AdamW(
        _param_groups(model, train_config.weight_decay),
        lr=train_config.lr,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / train_config.grad_accum))
    total_steps = steps_per_epoch * train_config.epochs
    warmup_steps = int(total_steps * train_config.warmup_ratio)
    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError(
            "O treino do Transformer precisa de transformers. Instale requirements.txt na máquina de treino."
        ) from exc
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    use_amp = bool(train_config.fp16) and device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = float("-inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    history: list[dict[str, object]] = []
    checkpoint_path: Path | None = None
    if output_dir is not None:
        checkpoint_path = Path(output_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            labels = batch.pop("labels")
            if use_amp:
                try:
                    autocast = torch.amp.autocast(device_type="cuda", enabled=True)
                except (TypeError, AttributeError):
                    autocast = torch.cuda.amp.autocast(enabled=True)
            else:
                from contextlib import nullcontext

                autocast = nullcontext()
            with autocast:
                logits = model(batch)
                loss = criterion(logits, labels) / train_config.grad_accum
            scaler.scale(loss).backward()
            running += float(loss.detach().item()) * train_config.grad_accum * int(labels.size(0))
            seen += int(labels.size(0))
            last_in_epoch = step == len(train_loader)
            if step % train_config.grad_accum == 0 or last_in_epoch:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = running / seen if seen else 0.0
        val_pred = predict(
            model,
            tokenizer,
            val,
            batch_size=train_config.batch_size,
            max_length=train_config.max_length,
            device=device,
        )
        val_gold = [example.label for example in val]
        report = evaluate(val_gold, val_pred)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": report.macro_f1,
            "val_macro_precision": report.macro_precision,
            "val_macro_recall": report.macro_recall,
        }
        history.append(row)
        if report.macro_f1 > best_f1:
            best_f1 = report.macro_f1
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path / "best.pt")
        else:
            stale += 1
            if stale >= train_config.patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return TrainResult(
        model=model,
        tokenizer=tokenizer,
        best_epoch=best_epoch,
        history=history,
        encoder_name=encoder_name,
        config=train_config,
    )
