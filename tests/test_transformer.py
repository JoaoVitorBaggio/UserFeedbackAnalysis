from pathlib import Path

import pytest

from user_feedback_analysis.data.load import load_jsonl
from user_feedback_analysis.data.split import SplitConfig, make_splits
from user_feedback_analysis.labels import LABELS, Label
from user_feedback_analysis.records import Example
from user_feedback_analysis.transformer.dataset import LABEL_TO_ID, SatisfactionDataset
from user_feedback_analysis.transformer.loss import LOSS_NAME, N_CLASSES, build_loss
from user_feedback_analysis.transformer.export import export_for_hub, load_exported
from user_feedback_analysis.transformer.model import DEFAULT_ENCODER, SatisfactionModel
from user_feedback_analysis.transformer.run import main
from user_feedback_analysis.transformer.train import TrainConfig, predict, train_transformer


def _tiny_bert(tmp_path: Path):
    transformers = pytest.importorskip("transformers")
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = [
        "thanks",
        "perfect",
        "great",
        "wrong",
        "fix",
        "translate",
        "write",
        "poem",
        "list",
        "work",
        "failed",
        "broken",
        "this",
        "is",
        "the",
        "a",
        "to",
        "it",
        "not",
        "answer",
    ]
    vocab = tmp_path / "vocab.txt"
    tokens = special + words + [f"tok{i}" for i in range(32)]
    vocab.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(vocab_file=str(vocab), do_lower_case=True)
    config = transformers.BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer, config


def _example(example_id: str, text: str, label: Label) -> Example:
    return Example(
        example_id=example_id,
        conversation_id=0,
        text=text,
        label=label,
        role="User",
        source_index=0,
    )


def test_label_ids_are_the_three_classes():
    assert N_CLASSES == 3
    assert LOSS_NAME == "categorical_cross_entropy"
    assert set(LABEL_TO_ID) == set(LABELS)
    assert set(LABEL_TO_ID.values()) == {0, 1, 2}
    assert LABEL_TO_ID[Label.SAT] == 0
    assert LABEL_TO_ID[Label.DSAT] == 1
    assert LABEL_TO_ID[Label.NO_SIGNAL] == 2
    assert DEFAULT_ENCODER == "bert-base-multilingual-cased"


def test_dataset_yields_text_and_index():
    example = _example("r0", "Thanks, this is perfect.", Label.SAT)
    dataset = SatisfactionDataset([example])
    assert len(dataset) == 1
    item = dataset[0]
    assert item["text"] == example.text
    assert item["label"] == 0
    assert item["label_name"] == "SAT"


def test_loss_helper_rejects_class_weights_before_training():
    with pytest.raises(ValueError, match="não pondera"):
        build_loss(class_weights=[1.0, 1.0, 1.0])


def test_empty_encoder_name_is_rejected():
    with pytest.raises(ValueError, match="vazio"):
        SatisfactionModel("")


def test_build_loss_is_unweighted_cross_entropy():
    torch = pytest.importorskip("torch")
    criterion = build_loss()
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.weight is None


def test_forward_returns_three_logits(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    tokenizer, config = _tiny_bert(tmp_path)
    model = SatisfactionModel("tiny-test", config=config)
    encoded = tokenizer(
        ["thanks perfect", "this is wrong"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    logits = model(encoded)
    assert tuple(logits.shape) == (2, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_training_rejects_empty_validation(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    tokenizer, config = _tiny_bert(tmp_path)
    train = [_example("t0", "thanks perfect", Label.SAT)]
    with pytest.raises(ValueError, match="validação"):
        train_transformer(
            train,
            [],
            encoder_name="tiny-test",
            seed=0,
            config=TrainConfig(epochs=1, batch_size=1, max_length=16, device="cpu", fp16=False),
            tokenizer=tokenizer,
            model_config=config,
        )


def test_one_epoch_on_fixture_yields_valid_labels(tiny_path: Path, tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    tokenizer, config = _tiny_bert(tmp_path)
    split = make_splits(
        load_jsonl(tiny_path),
        SplitConfig(seed=0, test_size=3, per_class=1, val_size=3, split_unit="turn"),
    )
    assert split.train and split.val and split.test
    result = train_transformer(
        split.train,
        split.val,
        encoder_name="tiny-test",
        seed=0,
        config=TrainConfig(
            epochs=1,
            batch_size=2,
            max_length=16,
            device="cpu",
            fp16=False,
            lr=1e-3,
            patience=1,
        ),
        tokenizer=tokenizer,
        model_config=config,
    )
    assert result.history
    assert result.best_epoch == 1
    assert "val_macro_f1" in result.history[0]
    assert 0.0 <= float(result.history[0]["val_macro_f1"]) <= 1.0
    predictions = predict(
        result.model,
        result.tokenizer,
        split.test,
        batch_size=2,
        max_length=16,
        device="cpu",
    )
    assert len(predictions) == len(split.test)
    assert all(isinstance(label, Label) for label in predictions)
    assert set(predictions) <= set(LABELS)


def test_same_seed_is_deterministic_on_cpu(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    tokenizer, config = _tiny_bert(tmp_path)
    train = [
        _example("tr0", "thanks perfect", Label.SAT),
        _example("tr1", "this is wrong", Label.DSAT),
        _example("tr2", "write a poem", Label.NO_SIGNAL),
        _example("tr3", "great answer", Label.SAT),
    ]
    val = [
        _example("va0", "thanks work", Label.SAT),
        _example("va1", "failed badly", Label.DSAT),
        _example("va2", "list the answer", Label.NO_SIGNAL),
    ]
    holdout = [
        _example("te0", "perfect work", Label.SAT),
        _example("te1", "broken fix", Label.DSAT),
        _example("te2", "translate this", Label.NO_SIGNAL),
    ]
    config_train = TrainConfig(
        epochs=1,
        batch_size=2,
        max_length=16,
        device="cpu",
        fp16=False,
        lr=1e-3,
        patience=1,
    )

    def _run():
        result = train_transformer(
            train,
            val,
            encoder_name="tiny-test",
            seed=0,
            config=config_train,
            tokenizer=tokenizer,
            model_config=config,
        )
        return predict(
            result.model,
            result.tokenizer,
            holdout,
            batch_size=2,
            max_length=16,
            device="cpu",
        )

    first = _run()
    second = _run()
    assert first == second
    assert set(first) <= set(LABELS)


def test_run_requires_val_size(tiny_path: Path, capsys):
    code = main(
        [
            "--data",
            str(tiny_path),
            "--test-size",
            "3",
            "--per-class",
            "1",
            "--seed",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert code == 1
    assert "val-size" in capsys.readouterr().err


def test_export_reload_keeps_predictions(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    tokenizer, config = _tiny_bert(tmp_path)
    train = [
        _example("tr0", "thanks perfect", Label.SAT),
        _example("tr1", "this is wrong", Label.DSAT),
        _example("tr2", "write a poem", Label.NO_SIGNAL),
        _example("tr3", "great answer", Label.SAT),
    ]
    val = [
        _example("va0", "thanks work", Label.SAT),
        _example("va1", "failed badly", Label.DSAT),
        _example("va2", "list the answer", Label.NO_SIGNAL),
    ]
    holdout = [
        _example("te0", "perfect work", Label.SAT),
        _example("te1", "broken fix", Label.DSAT),
        _example("te2", "translate this", Label.NO_SIGNAL),
    ]
    result = train_transformer(
        train,
        val,
        encoder_name="tiny-test",
        seed=0,
        config=TrainConfig(
            epochs=1,
            batch_size=2,
            max_length=16,
            device="cpu",
            fp16=False,
            lr=1e-3,
            patience=1,
        ),
        tokenizer=tokenizer,
        model_config=config,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(result.model.state_dict(), checkpoint)
    before = predict(
        result.model,
        result.tokenizer,
        holdout,
        batch_size=2,
        max_length=16,
        device="cpu",
    )
    dest = tmp_path / "export"
    export_for_hub(
        checkpoint,
        dest,
        encoder_name="tiny-test",
        config=config,
        tokenizer=tokenizer,
        run_record={
            "official_result": False,
            "metrics": {"macro_f1": 0.5, "n": 3, "per_class": {}},
            "training": {"best_epoch": 1, "best_val_macro_f1": 0.4},
            "split": {"seed": 0},
        },
    )
    assert (dest / "classifier_head.safetensors").is_file()
    assert (dest / "label_map.json").is_file()
    restored, restored_tok = load_exported(dest)
    after = predict(
        restored,
        restored_tok,
        holdout,
        batch_size=2,
        max_length=16,
        device="cpu",
    )
    assert after == before
