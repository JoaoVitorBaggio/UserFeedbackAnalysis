import json
from pathlib import Path

import pytest

from user_feedback_analysis.data.load import load_jsonl
from user_feedback_analysis.data.split import SplitConfig, make_splits
from user_feedback_analysis.harness import evaluate
from user_feedback_analysis.labels import Label
from user_feedback_analysis.llm.client import classify_text, resolve_dry_run
from user_feedback_analysis.llm.parse import parse_label
from user_feedback_analysis.llm.prompts import build_prompt, clip_text
from user_feedback_analysis.llm.run import main
from user_feedback_analysis.llm.sample import sample_demonstrations


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SAT", Label.SAT),
        ("dsat", Label.DSAT),
        ("sem sinal", Label.NO_SIGNAL),
        ("The label is DSAT.", Label.DSAT),
        ("no-signal", Label.NO_SIGNAL),
    ],
)
def test_parse_label(text, expected):
    assert parse_label(text) is expected


def test_parse_rejects_two_labels_and_empty_text():
    with pytest.raises(ValueError):
        parse_label("SAT and DSAT")
    with pytest.raises(ValueError):
        parse_label("hello")


def test_zero_shot_prompt_has_no_examples():
    prompt, truncated = build_prompt("Translate this paragraph.")
    assert "Example" not in prompt
    assert "Translate this paragraph." in prompt
    assert "SAT" in prompt and "DSAT" in prompt and "sem sinal" in prompt
    assert truncated == 0


def test_clip_is_visible_in_the_prompt():
    prompt, truncated = build_prompt("abcdef", max_chars=3)
    assert truncated == 1
    assert "abc" in prompt
    assert "abcdef" not in prompt
    assert clip_text("abcdef", 3) == ("abc", True)


def test_few_shot_examples_come_from_train_only(tiny_path: Path):
    corpus = load_jsonl(tiny_path)
    split = make_splits(corpus, SplitConfig(seed=0, test_size=3, per_class=1))
    demos = sample_demonstrations(split.train, k=3, seed=0)
    again = sample_demonstrations(split.train, k=3, seed=0)
    assert [item.example_id for item in demos] == [item.example_id for item in again]
    assert {item.example_id for item in demos} <= {item.example_id for item in split.train}
    assert {item.example_id for item in demos}.isdisjoint(
        {item.example_id for item in split.test}
    )
    assert {item.label for item in demos} == {Label.SAT, Label.DSAT, Label.NO_SIGNAL}
    prompt, _ = build_prompt(split.test[0].text, demos)
    assert "Example 1" in prompt
    assert split.test[0].text in prompt
    for demo in demos:
        assert demo.text in prompt
        assert f"Label: {demo.label.value}" in prompt


def test_dry_run_resolution(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert resolve_dry_run(None) is True
    assert resolve_dry_run(True) is True
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        resolve_dry_run(None)
    monkeypatch.setenv("LLM_MODEL", "dummy-model")
    assert resolve_dry_run(None) is False
    assert resolve_dry_run(True) is True


def test_live_client_parses_a_fake_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "dummy-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9/v1/")
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "The label is DSAT."}}]}
            ).encode("utf-8")

    def _urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    label = classify_text(
        "This is still wrong, fix it.",
        [],
        dry_run=False,
        model="dummy-model",
        max_chars=100,
        timeout=5,
    )
    assert label is Label.DSAT
    assert captured["url"] == "http://localhost:9/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["model"] == "dummy-model"
    assert "sk-test" not in captured["body"]["messages"][1]["content"]


def test_llm_dry_run_entrypoint(tiny_path: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "should-not-be-called")

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    output = tmp_path / "llm.json"
    code = main(
        [
            "--data",
            str(tiny_path),
            "--shot",
            "few",
            "--k",
            "3",
            "--test-size",
            "3",
            "--per-class",
            "1",
            "--seed",
            "0",
            "--dry-run",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["arm"] == "llm_few_shot"
    assert payload["dry_run"] is True
    assert payload["official_result"] is False
    assert payload["model"] == "mock"
    assert payload["k"] == 3
    assert set(payload["demonstration_ids"]).isdisjoint(payload["split"]["test_ids"])
    assert set(payload["demonstration_ids"]) <= set(payload["split"]["train_ids"])
    report = evaluate(
        [row["gold"] for row in payload["predictions"]],
        [row["pred"] for row in payload["predictions"]],
    )
    assert report.macro_f1 == pytest.approx(payload["metrics"]["macro_f1"])


def test_zero_shot_dry_run_entrypoint(tiny_path: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    output = tmp_path / "zero.json"
    code = main(
        [
            "--data",
            str(tiny_path),
            "--shot",
            "zero",
            "--test-size",
            "3",
            "--per-class",
            "1",
            "--seed",
            "0",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["arm"] == "llm_zero_shot"
    assert payload["dry_run"] is True
    assert payload["k"] == 0
    assert payload["demonstration_ids"] == []


def test_zero_shot_rejects_k(tiny_path: Path):
    code = main(
        [
            "--data",
            str(tiny_path),
            "--shot",
            "zero",
            "--k",
            "2",
            "--test-size",
            "3",
            "--per-class",
            "1",
            "--seed",
            "0",
            "--dry-run",
        ]
    )
    assert code == 1
