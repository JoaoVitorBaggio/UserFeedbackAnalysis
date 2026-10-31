import json
import sys
from pathlib import Path

import pytest

from user_feedback_analysis.baseline.mapping import ALTERNATIVE_MODEL_ID, mapping_dict
from user_feedback_analysis.baseline.run import main
from user_feedback_analysis.harness import evaluate


def test_mock_entrypoint_scores_with_the_harness(tiny_path: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert "transformers" not in sys.modules
    output = tmp_path / "baseline.json"
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
            "--mock",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert "transformers" not in sys.modules
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["arm"] == "baseline"
    assert payload["official_result"] is False
    assert payload["classifier"]["kind"] == "mock-lexicon"
    assert payload["classifier"]["model_id"] is None
    assert payload["classifier"]["mapping"] == mapping_dict()
    assert payload["classifier"]["mapping"] == {
        "positive": "SAT",
        "negative": "DSAT",
        "neutral": "sem sinal",
    }
    assert set(payload["metrics"]["labels"]) == {"SAT", "DSAT", "sem sinal"}
    report = evaluate(
        [row["gold"] for row in payload["predictions"]],
        [row["pred"] for row in payload["predictions"]],
    )
    assert report.macro_f1 == pytest.approx(payload["metrics"]["macro_f1"])
    assert 0.0 <= payload["metrics"]["macro_f1"] <= 1.0
    test_ids = set(payload["split"]["test_ids"])
    train_ids = set(payload["split"]["train_ids"])
    assert test_ids.isdisjoint(train_ids)
    assert payload["split"]["per_class_actual"] == {"SAT": 1, "DSAT": 1, "sem sinal": 1}


def test_protocol_fraction_is_the_cli_default(tiny_path: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    output = tmp_path / "fraction.json"
    code = main(
        [
            "--data",
            str(tiny_path),
            "--per-class",
            "1",
            "--seed",
            "0",
            "--split-unit",
            "turn",
            "--mock",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert "transformers" not in sys.modules
    payload = json.loads(output.read_text(encoding="utf-8"))
    split = payload["split"]
    assert split["stratified"] is True
    assert split["test_fraction_requested"] == 0.12
    assert split["test_size_requested"] is None
    assert split["seed"] == 0
    assert split["split_unit"] == "turn"
    assert split["test_class_targets"] == {"SAT": 1, "DSAT": 0, "sem sinal": 0}
    assert split["test_class_counts"] == {"SAT": 1, "DSAT": 0, "sem sinal": 0}
    assert set(split["test_ids"]).isdisjoint(split["train_ids"])


def test_checkpoint_override_stays_offline_in_mock(tiny_path: Path, tmp_path: Path):
    output = tmp_path / "checkpoint.json"
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
            "--mock",
            "--checkpoint",
            ALTERNATIVE_MODEL_ID,
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert "transformers" not in sys.modules
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classifier"]["kind"] == "mock-lexicon"
    assert payload["classifier"]["model_id"] is None


def test_mock_lexicon_uses_the_sentiment_map():
    from user_feedback_analysis.baseline.mock import LexiconMockClassifier
    from user_feedback_analysis.labels import Label

    labels = LexiconMockClassifier().predict(
        [
            "Thanks, this is perfect.",
            "This is still wrong, fix it.",
            "List three colors.",
        ]
    )
    assert labels == [Label.SAT, Label.DSAT, Label.NO_SIGNAL]
