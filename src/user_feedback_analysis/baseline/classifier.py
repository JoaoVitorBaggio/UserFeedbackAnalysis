"""Checkpoint Hugging Face de sentimento, com as três classes.

Inferência do RoBERTa-base cabe em CPU. Na GTX 1060 6GB, fp32 com batch pequeno
costuma caber; o pico de VRAM não foi medido neste repositório. Se a placa
recusar, use CPU ou baixe ``batch_size``.
"""

from __future__ import annotations

from user_feedback_analysis.baseline.mapping import (
    DEFAULT_MODEL_ID,
    pipeline_label_to_sentiment,
    sentiment_to_label,
    validate_id2label,
)
from user_feedback_analysis.labels import Label

_TWEET_MODEL_PREFIX = "cardiffnlp/twitter-roberta-base-sentiment"


def preprocess_cardiff_twitter(text: str) -> str:
    """Pré-processamento do card do modelo: @menção → @user, URL → http."""
    tokens: list[str] = []
    for token in text.split(" "):
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        elif token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens)


def uses_tweet_preprocess(model_id: str) -> bool:
    return model_id.startswith(_TWEET_MODEL_PREFIX)


class HuggingFaceSentimentClassifier:
    kind = "huggingface"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size é um inteiro >= 1.")
        if max_length < 1:
            raise ValueError("max_length é um inteiro >= 1.")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.preprocess = uses_tweet_preprocess(model_id)
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "O baseline real precisa de transformers e torch (requirements.txt)."
            ) from exc
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA indisponível. Rode com --device cpu. "
                    "Com a GTX 1060 6GB presente, use --device cuda e batch pequeno."
                )
        device_id = _pipeline_device(device)
        model_kwargs = {"local_files_only": True} if local_files_only else {}
        tokenizer_kwargs = {"local_files_only": True} if local_files_only else {}
        self._pipe = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            device=device_id,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self._id2label = validate_id2label(
            dict(self._pipe.model.config.id2label),
            model_id,
        )

    def predict(self, texts: list[str]) -> list[Label]:
        if not texts:
            return []
        prepared = [
            preprocess_cardiff_twitter(text) if self.preprocess else text for text in texts
        ]
        if any(not text.strip() for text in prepared):
            raise ValueError("O baseline recebeu texto vazio.")
        outputs = self._pipe(
            prepared,
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )
        labels: list[Label] = []
        for output in outputs:
            raw = _top_label(output)
            sentiment = pipeline_label_to_sentiment(raw, self._id2label)
            labels.append(sentiment_to_label(sentiment))
        if len(labels) != len(texts):
            raise RuntimeError("O checkpoint devolveu um número inesperado de rótulos.")
        return labels


def _pipeline_device(device: str) -> int:
    key = device.strip().casefold()
    if key == "cpu":
        return -1
    if key == "cuda":
        return 0
    raise ValueError("device é 'cpu' ou 'cuda'.")


def _top_label(output: object) -> str:
    item = output
    if isinstance(item, list):
        if not item:
            raise RuntimeError("O checkpoint devolveu uma lista vazia.")
        item = item[0]
    if not isinstance(item, dict) or "label" not in item:
        raise RuntimeError(f"Saída inesperada do pipeline: {output!r}.")
    return str(item["label"])
