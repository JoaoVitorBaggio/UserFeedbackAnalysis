"""Checkpoint e retomada da corrida de LLM no mesmo --output."""

import json
import os
from pathlib import Path

import pytest

from user_feedback_analysis.harness import write_json
from user_feedback_analysis.labels import Label
from user_feedback_analysis.llm.checkpoint import (
    CHECKPOINT_KIND,
    build_fingerprint,
    load_resume,
    write_checkpoint,
)
from user_feedback_analysis.llm.client import VoteResult, classify_votes
from user_feedback_analysis.llm.protocol import (
    PROTOCOL_LLM_BASE_URL,
    PROTOCOL_LLM_MODEL_ID,
    llm_official_result,
)
from user_feedback_analysis.llm.run import main
from user_feedback_analysis.records import Example


def _example(index: int, text: str, label: Label) -> Example:
    return Example(
        example_id=f"r{index}",
        conversation_id=index,
        text=text,
        label=label,
        role="User",
        source_index=index,
    )


def _fingerprint(examples: list[Example], **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "shot": "zero",
        "seed": 0,
        "few_shot_seed": None,
        "few_shot_k": 0,
        "inferences": 1,
        "temperature": 0.7,
        "max_chars": 4000,
        "dry_run": True,
        "model_id": None,
        "base_url": "https://api.openai.com/v1",
        "test_subsample": None,
        "test_fraction": None,
        "test_size": 3,
        "per_class": 1,
        "strict_per_class": False,
        "split_unit": "conversation",
        "role": "User",
        "val_size": None,
        "demonstrations": [],
        "evaluated": examples,
    }
    kwargs.update(overrides)
    return build_fingerprint(**kwargs)  # type: ignore[arg-type]


def _vote(label: Label) -> VoteResult:
    return VoteResult(label=label, votes=(label,), decision="majority")


def _isolate(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)


def _zero_args(tiny_path: Path, output: Path, *, seed: str = "0", shot: str = "zero") -> list[str]:
    args = [
        "--data",
        str(tiny_path),
        "--shot",
        shot,
        "--test-size",
        "3",
        "--per-class",
        "1",
        "--seed",
        seed,
        "--dry-run",
        "--inferences",
        "1",
        "--output",
        str(output),
    ]
    if shot == "few":
        args[4:4] = ["--k", "3"]
    return args


def _counting_votes(monkeypatch, calls: list[str], *, fail_at: int | None = None):
    real = classify_votes

    def wrapped(text, demonstrations, **kwargs):
        calls.append(text)
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError("queda no meio da semente")
        return real(text, demonstrations, **kwargs)

    monkeypatch.setattr("user_feedback_analysis.llm.run.classify_votes", wrapped)


def test_checkpoint_roundtrip_reuses_votes_and_hides_the_partial_result(tmp_path: Path):
    examples = [
        _example(0, "Thanks, this is perfect.", Label.SAT),
        _example(1, "This is still wrong, fix it.", Label.DSAT),
    ]
    fingerprint = _fingerprint(examples)
    path = tmp_path / "run.json"
    write_checkpoint(path, fingerprint, examples, [_vote(Label.SAT)])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["kind"] == CHECKPOINT_KIND
    assert saved["official_result"] is False
    assert saved["next_index"] == 1
    assert saved["n_completed"] == 1
    assert saved["n_expected"] == 2
    assert saved["complete"] is False
    resumed = load_resume(path, fingerprint, examples)
    assert len(resumed) == 1
    assert resumed[0].label is Label.SAT
    assert resumed[0].votes == (Label.SAT,)


def test_load_resume_rejects_a_different_corpus_without_deleting(tmp_path: Path):
    examples = [_example(0, "Thanks, this is perfect.", Label.SAT)]
    path = tmp_path / "run.json"
    fingerprint = _fingerprint(examples)
    write_checkpoint(path, fingerprint, examples, [_vote(Label.SAT)])
    before = path.read_text(encoding="utf-8")
    changed = [_example(0, "outro texto", Label.SAT)]
    with pytest.raises(ValueError, match="evaluated_digest"):
        load_resume(path, _fingerprint(changed), changed)
    assert path.read_text(encoding="utf-8") == before

    tampered = json.loads(before)
    tampered["completed"][0]["gold"] = "DSAT"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    held = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="ouro"):
        load_resume(path, fingerprint, examples)
    assert path.read_text(encoding="utf-8") == held


def test_load_resume_rejects_truncated_and_foreign_files(tmp_path: Path):
    examples = [_example(0, "Thanks, this is perfect.", Label.SAT)]
    fingerprint = _fingerprint(examples)
    truncated = tmp_path / "trunc.json"
    truncated.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="não fecha"):
        load_resume(truncated, fingerprint, examples)
    assert truncated.read_text(encoding="utf-8") == "{"

    foreign = tmp_path / "other.json"
    foreign.write_text('{"arm": "baseline", "official_result": true}\n', encoding="utf-8")
    before = foreign.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="não é uma corrida de LLM"):
        load_resume(foreign, fingerprint, examples)
    assert foreign.read_text(encoding="utf-8") == before
    assert load_resume(tmp_path / "ausente.json", fingerprint, examples) == ()


def test_write_json_keeps_the_previous_file_when_replace_fails(tmp_path: Path, monkeypatch):
    path = tmp_path / "run.json"
    write_json({"intact": True, "next_index": 1}, path)

    def boom(src, dst):
        raise OSError("disco cheio")

    monkeypatch.setattr("user_feedback_analysis.harness.os.replace", boom)
    with pytest.raises(OSError, match="disco cheio"):
        write_json({"intact": False, "next_index": 2}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"intact": True, "next_index": 1}
    assert list(tmp_path.glob(".run.json.*.tmp")) == []


def test_dry_run_resume_skips_finished_examples_and_keeps_official_result(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    _isolate(monkeypatch)
    output = tmp_path / "zero.json"
    calls: list[str] = []
    _counting_votes(monkeypatch, calls, fail_at=2)
    assert main(_zero_args(tiny_path, output)) == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["kind"] == CHECKPOINT_KIND
    assert saved["next_index"] == 1
    assert saved["official_result"] is False
    assert "sk-" not in output.read_text(encoding="utf-8")
    n_expected = saved["n_expected"]
    calls.clear()

    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output)) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "kind" not in payload
    assert payload["resume"]["n_reused"] == 1
    assert payload["official_result"] is False
    assert payload["dry_run"] is True
    assert len(calls) == n_expected - 1
    deviations = payload["protocol_deviations"]
    assert any(item.startswith("dry_run") for item in deviations)
    assert all("checkpoint" not in item and "retom" not in item for item in deviations)

    fresh_calls: list[str] = []
    _counting_votes(monkeypatch, fresh_calls)
    fresh = tmp_path / "fresh.json"
    assert main(_zero_args(tiny_path, fresh)) == 0
    fresh_payload = json.loads(fresh.read_text(encoding="utf-8"))
    assert fresh_payload["protocol_deviations"] == deviations
    assert fresh_payload["official_result"] is payload["official_result"]
    assert fresh_payload["predictions"] == payload["predictions"]
    assert fresh_payload["inference_votes"] == payload["inference_votes"]
    assert fresh_payload["metrics"] == payload["metrics"]
    assert fresh_payload["resume"]["n_reused"] == 0
    assert len(fresh_calls) == n_expected

    calls.clear()
    assert main(_zero_args(tiny_path, output)) == 0
    again = json.loads(output.read_text(encoding="utf-8"))
    assert calls == []
    assert again["resume"]["n_reused"] == n_expected
    assert again["predictions"] == payload["predictions"]
    assert again["official_result"] is False

    closed, closed_deviations = llm_official_result(
        dry_run=False,
        model_id=PROTOCOL_LLM_MODEL_ID,
        base_url=PROTOCOL_LLM_BASE_URL,
        shot="few",
        few_shot_k=3,
        inferences=3,
        test_subsample=1000,
        test_fraction=0.12,
        per_class=5000,
        strict_per_class=True,
        seed=1,
        fallback_base_url="https://api.openai.com/v1",
    )
    assert closed is True
    assert closed_deviations == []


def test_resume_refuses_another_seed_and_does_not_clobber(
    tiny_path: Path, tmp_path: Path, monkeypatch, capsys
):
    _isolate(monkeypatch)
    output = tmp_path / "zero.json"
    calls: list[str] = []
    _counting_votes(monkeypatch, calls, fail_at=2)
    assert main(_zero_args(tiny_path, output)) == 1
    before = output.read_bytes()
    calls.clear()
    code = main(_zero_args(tiny_path, output, seed="1"))
    captured = capsys.readouterr()
    assert code == 1
    assert "seed" in captured.err
    assert "Não apaguei" in captured.err
    assert output.read_bytes() == before
    assert calls == []


def test_failed_replace_keeps_the_last_good_example(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    _isolate(monkeypatch)
    output = tmp_path / "zero.json"
    calls: list[str] = []
    _counting_votes(monkeypatch, calls, fail_at=2)
    assert main(_zero_args(tiny_path, output)) == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["next_index"] == 1
    done = saved["completed"][0]["id"]
    calls.clear()

    def boom(src, dst):
        raise OSError("disco cheio")

    # harness.os é o módulo os. Guardar a função antes do patch.
    original_replace = os.replace
    monkeypatch.setattr("user_feedback_analysis.harness.os.replace", boom)
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output)) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["next_index"] == 1
    assert calls  # o exemplo seguinte foi inferido, mas o replace não entrou
    lost = calls[0]
    assert list(tmp_path.glob(".zero.json.*.tmp")) == []

    monkeypatch.setattr("user_feedback_analysis.harness.os.replace", original_replace)
    calls.clear()
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output)) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["resume"]["n_reused"] == 1
    assert payload["predictions"][0]["id"] == done
    assert len(calls) == saved["n_expected"] - 1
    assert lost in calls


def test_crash_before_the_final_json_still_resumes(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    _isolate(monkeypatch)
    output = tmp_path / "zero.json"
    calls: list[str] = []
    real_write = write_json

    def fail_final(payload, path):
        if payload.get("arm") in {"llm_zero_shot", "llm_few_shot"}:
            raise OSError("queda antes do JSON final")
        real_write(payload, path)

    monkeypatch.setattr("user_feedback_analysis.llm.checkpoint.write_json", fail_final)
    monkeypatch.setattr("user_feedback_analysis.llm.run.write_json", fail_final)
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output)) == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["kind"] == CHECKPOINT_KIND
    assert saved["complete"] is True
    assert saved["next_index"] == saved["n_expected"]
    assert saved["official_result"] is False
    n_expected = saved["n_expected"]
    assert len(calls) == n_expected

    monkeypatch.setattr("user_feedback_analysis.llm.checkpoint.write_json", real_write)
    monkeypatch.setattr("user_feedback_analysis.llm.run.write_json", real_write)
    calls.clear()
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output)) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert calls == []
    assert "kind" not in payload
    assert payload["resume"]["n_reused"] == n_expected
    assert payload["official_result"] is False


def test_legacy_final_without_fingerprint_is_reused(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    _isolate(monkeypatch)
    output = tmp_path / "zero.json"
    calls: list[str] = []
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output)) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    del payload["run_fingerprint"]
    output.write_text(json.dumps(payload), encoding="utf-8")
    calls.clear()
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output)) == 0
    again = json.loads(output.read_text(encoding="utf-8"))
    assert calls == []
    assert again["predictions"] == payload["predictions"]
    assert again["official_result"] is False


def test_few_shot_rerun_does_not_reclassify(tiny_path: Path, tmp_path: Path, monkeypatch):
    _isolate(monkeypatch)
    output = tmp_path / "few.json"
    calls: list[str] = []
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output, shot="few")) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["k"] == 3
    assert payload["demonstration_ids"]
    assert payload["official_result"] is False
    assert "sk-" not in output.read_text(encoding="utf-8")
    assert len(calls) == len(payload["predictions"])
    calls.clear()
    _counting_votes(monkeypatch, calls)
    assert main(_zero_args(tiny_path, output, shot="few")) == 0
    assert calls == []
    again = json.loads(output.read_text(encoding="utf-8"))
    assert again["demonstration_ids"] == payload["demonstration_ids"]
    assert again["predictions"] == payload["predictions"]
    assert again["resume"]["n_reused"] == len(payload["predictions"])


def test_checkpoint_does_not_store_a_key_pasted_into_the_base_url(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    _isolate(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://sk-super-secret@api.deepinfra.com/v1/openai",
    )
    output = tmp_path / "zero.json"
    calls: list[str] = []
    _counting_votes(monkeypatch, calls, fail_at=2)
    assert main(_zero_args(tiny_path, output)) == 1
    text = output.read_text(encoding="utf-8")
    assert "sk-super-secret" not in text
    saved = json.loads(text)
    assert saved["fingerprint"]["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert saved["official_result"] is False
