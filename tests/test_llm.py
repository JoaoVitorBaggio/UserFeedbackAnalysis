import http.client
import io
import json
import ssl
import urllib.error
from email.message import EmailMessage
from pathlib import Path

import pytest

from user_feedback_analysis.data.load import load_jsonl
from user_feedback_analysis.data.split import SplitConfig, make_splits
from user_feedback_analysis.harness import evaluate
from user_feedback_analysis.labels import Label
from user_feedback_analysis.llm.client import (
    DEFAULT_BASE_URL,
    LLM_MAX_ATTEMPTS,
    LLM_USER_AGENT,
    _RETRY_CAP_S,
    chat_complete,
    classify_text,
    classify_votes,
    resolve_dry_run,
)
from user_feedback_analysis.llm.protocol import (
    PROTOCOL_LLM_MODEL_ID,
    llm_official_result,
    public_base_url,
)
from user_feedback_analysis.llm.vote import majority_vote
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
    user_agents: list[str | None] = []

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
        captured["calls"] = int(captured.get("calls", 0)) + 1
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        # urllib guarda o cabeçalho como "User-agent".
        user_agents.append(request.get_header("User-agent"))
        return _Response()

    def _no_sleep(seconds: float) -> None:
        raise AssertionError(f"chamada bem-sucedida não espera ({seconds})")

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", _no_sleep)
    label = classify_text(
        "This is still wrong, fix it.",
        [],
        dry_run=False,
        model="dummy-model",
        max_chars=100,
        timeout=5,
    )
    assert label is Label.DSAT
    assert captured["calls"] == 3
    assert captured["url"] == "http://localhost:9/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert user_agents == [LLM_USER_AGENT] * 3
    assert LLM_USER_AGENT.startswith("Mozilla/5.0")
    assert "UserFeedbackAnalysis/llm-client" in LLM_USER_AGENT
    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["model"] == "dummy-model"
    assert "sk-test" not in captured["body"]["messages"][1]["content"]
    classify_text(
        "This is still wrong, fix it.",
        [],
        dry_run=False,
        model="dummy-model",
        max_chars=100,
        timeout=5,
        k=1,
        temperature=0,
    )
    assert captured["body"]["temperature"] == 0
    assert captured["calls"] == 4
    assert user_agents == [LLM_USER_AGENT] * 4


def _http_error(
    code: int,
    body: str,
    headers: EmailMessage | None = None,
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://localhost:9/v1/chat/completions",
        code,
        "error",
        headers if headers is not None else EmailMessage(),
        io.BytesIO(body.encode("utf-8")),
    )


def _ok_response(content: str = "SAT"):
    encoded = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return encoded

    return _Response()


def _install_urlopen(monkeypatch, outcomes: list[object]) -> dict[str, object]:
    seen: dict[str, object] = {"calls": 0, "agents": [], "auths": []}

    def _urlopen(request, timeout=0):
        seen["calls"] = int(seen["calls"]) + 1
        seen["agents"].append(request.get_header("User-agent"))
        seen["auths"].append(request.get_header("Authorization"))
        seen["timeout"] = timeout
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    return seen


def test_chat_complete_retries_429_using_the_server_hint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    body = (
        '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-20b` '
        'on tokens per minute (TPM): Limit 8000, Used 7600, Requested 950. '
        'Please try again in 1.5s.","type":"tokens","code":"rate_limit_exceeded"}}'
    )
    seen = _install_urlopen(
        monkeypatch,
        [_http_error(429, body), _ok_response("SAT")],
    )
    assert chat_complete("Classify.", model="openai/gpt-oss-20b", timeout=5) == "SAT"
    assert seen["calls"] == 2
    assert seen["agents"] == [LLM_USER_AGENT, LLM_USER_AGENT]
    assert seen["auths"] == ["Bearer sk-test", "Bearer sk-test"]
    assert seen["timeout"] == 5
    assert sleeps == [1.5]

    precise = _install_urlopen(
        monkeypatch,
        [_http_error(429, "Please try again in 5.289s."), _ok_response()],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="openai/gpt-oss-20b") == "SAT"
    assert precise["calls"] == 2
    assert sleeps == pytest.approx([5.289])

    headers = EmailMessage()
    headers["Retry-After"] = "4"
    longer_header = _install_urlopen(
        monkeypatch,
        [_http_error(429, "Please try again in 1.5s.", headers), _ok_response()],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="m") == "SAT"
    assert longer_header["calls"] == 2
    assert sleeps == [4.0]

    longer_body = EmailMessage()
    longer_body["Retry-After"] = "2"
    _install_urlopen(
        monkeypatch,
        [_http_error(429, "Please try again in 4.1s.", longer_body), _ok_response()],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="m") == "SAT"
    assert sleeps == pytest.approx([4.1])

    _install_urlopen(
        monkeypatch,
        [_http_error(429, "Please try again in 250ms."), _ok_response()],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="m") == "SAT"
    assert sleeps == pytest.approx([0.25])


def test_chat_complete_stops_retrying_429(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [_http_error(429, "Please try again in 9s.") for _ in range(LLM_MAX_ATTEMPTS)],
    )
    with pytest.raises(
        RuntimeError,
        match=rf"LLM HTTP 429:.*após {LLM_MAX_ATTEMPTS} tentativas",
    ):
        chat_complete("Classify.", model="openai/gpt-oss-20b")
    assert seen["calls"] == LLM_MAX_ATTEMPTS
    assert sleeps == [9.0] * (LLM_MAX_ATTEMPTS - 1)
    assert seen["agents"] == [LLM_USER_AGENT] * LLM_MAX_ATTEMPTS

    capped = _install_urlopen(
        monkeypatch,
        [_http_error(429, "Please try again in 10000s."), _ok_response()],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="m") == "SAT"
    assert capped["calls"] == 2
    assert sleeps == [_RETRY_CAP_S]


def test_chat_complete_retries_5xx_and_transient_network(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [
            _http_error(503, ""),
            _http_error(500, "upstream"),
            _ok_response("DSAT"),
        ],
    )
    assert chat_complete("Classify.", model="m", timeout=3) == "DSAT"
    assert seen["calls"] == 3
    assert sleeps == [1.0, 2.0]

    timed_out = _install_urlopen(
        monkeypatch,
        [TimeoutError("timed out"), _ok_response("SAT")],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="m") == "SAT"
    assert timed_out["calls"] == 2
    assert sleeps == [1.0]

    reset = _install_urlopen(
        monkeypatch,
        [urllib.error.URLError(ConnectionResetError("connection reset")), _ok_response()],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="m") == "SAT"
    assert reset["calls"] == 2
    assert sleeps == [1.0]

    partial = _install_urlopen(
        monkeypatch,
        [http.client.IncompleteRead(b"partial"), _ok_response()],
    )
    sleeps.clear()
    assert chat_complete("Classify.", model="m") == "SAT"
    assert partial["calls"] == 2
    assert sleeps == [1.0]


def test_chat_complete_does_not_retry_permanent_failures(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)

    cloudflare = _install_urlopen(
        monkeypatch,
        [_http_error(403, "error code: 1010")],
    )
    with pytest.raises(RuntimeError, match=r"LLM HTTP 403: error code: 1010$"):
        chat_complete("Classify.", model="m")
    assert cloudflare["calls"] == 1
    assert sleeps == []

    unauthorized = _install_urlopen(
        monkeypatch,
        [_http_error(401, '{"error":{"message":"Invalid API Key"}}')],
    )
    with pytest.raises(RuntimeError, match="LLM HTTP 401"):
        chat_complete("Classify.", model="m")
    assert unauthorized["calls"] == 1

    bad_url = _install_urlopen(
        monkeypatch,
        [urllib.error.URLError("unknown url type: ftp")],
    )
    with pytest.raises(RuntimeError, match="unknown url type: ftp$"):
        chat_complete("Classify.", model="m")
    assert bad_url["calls"] == 1

    cert = _install_urlopen(
        monkeypatch,
        [urllib.error.URLError(ssl.SSLError("certificate verify failed"))],
    )
    with pytest.raises(RuntimeError, match=r"certificate verify failed"):
        chat_complete("Classify.", model="m")
    assert cert["calls"] == 1
    assert sleeps == []


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
    assert payload["inference_k"] == 3
    assert payload["vote"] == "majority"
    assert payload["tie_break"] == "first_inference"
    assert payload["temperature"] == 0.7
    assert payload["n_tie_breaks"] == 0
    assert payload["split"]["stratified"] is False
    assert payload["split"]["seed"] == 0
    assert payload["live"] is False
    assert payload["test_subsample"]["applied"] is False
    assert payload["test_subsample"]["requested"] is None
    assert payload["test_subsample"]["official_test_size"] == payload["split"]["n_test"]
    assert payload["model_requirement"]["accepted_model_id"] == "openai/gpt-oss-20b"
    assert payload["model_requirement"]["revoked_status"] == "revoked"
    assert payload["model_id"] == "should-not-be-called"
    assert any(item.startswith("dry_run") for item in payload["protocol_deviations"])
    assert all(len(votes) == 3 for votes in payload["inference_votes"])
    assert len(payload["inference_votes"]) == len(payload["predictions"])
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


def test_majority_vote_and_first_inference_tie():
    assert majority_vote([Label.SAT, Label.DSAT, Label.DSAT]) == (Label.DSAT, "majority")
    assert majority_vote([Label.SAT, Label.SAT, Label.SAT]) == (Label.SAT, "majority")
    assert majority_vote([Label.DSAT, Label.SAT, Label.NO_SIGNAL]) == (
        Label.DSAT,
        "first_inference",
    )
    assert majority_vote([Label.DSAT, Label.SAT, Label.SAT, Label.DSAT]) == (
        Label.DSAT,
        "first_inference",
    )


def test_k3_uses_a_mock_client_and_does_not_touch_the_network():
    calls: list[float] = []

    def complete(prompt: str, *, model: str, timeout: float, temperature: float) -> str:
        calls.append(temperature)
        return ("SAT", "DSAT", "DSAT")[len(calls) - 1]

    voted = classify_votes(
        "List three colors.",
        [],
        dry_run=False,
        model="dummy-model",
        max_chars=100,
        k=3,
        temperature=0.4,
        complete=complete,
    )
    assert voted.label is Label.DSAT
    assert voted.decision == "majority"
    assert voted.votes == (Label.SAT, Label.DSAT, Label.DSAT)
    assert calls == [0.4, 0.4, 0.4]

    sequence = iter(("SAT", "DSAT", "sem sinal"))

    def tied(prompt: str, *, model: str, timeout: float, temperature: float) -> str:
        return next(sequence)

    tie = classify_votes(
        "List three colors.",
        [],
        dry_run=False,
        model="dummy-model",
        max_chars=None,
        k=3,
        temperature=0.7,
        complete=tied,
    )
    assert tie.label is Label.SAT
    assert tie.decision == "first_inference"

    def boom(*args, **kwargs):
        raise AssertionError("dry-run chamou o cliente")

    dry = classify_votes(
        "List three colors.",
        [],
        dry_run=True,
        model=None,
        max_chars=100,
        k=3,
        complete=boom,
    )
    assert dry.label is Label.NO_SIGNAL
    assert dry.votes == (Label.NO_SIGNAL, Label.NO_SIGNAL, Label.NO_SIGNAL)
    assert dry.decision == "majority"


def test_temperature_and_inference_k_land_in_the_run_json(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    output = tmp_path / "temp.json"
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
            "--dry-run",
            "--inferences",
            "1",
            "--temperature",
            "0.3",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["inference_k"] == 1
    assert payload["temperature"] == 0.3
    assert payload["tie_break"] == "first_inference"
    assert all(len(votes) == 1 for votes in payload["inference_votes"])


def _closed_llm_call(**overrides: object) -> tuple[bool, list[str]]:
    kwargs: dict[str, object] = {
        "dry_run": False,
        "model_id": PROTOCOL_LLM_MODEL_ID,
        "base_url": "https://api.groq.com/openai/v1/",
        "shot": "few",
        "few_shot_k": 3,
        "inferences": 3,
        "test_subsample": 1000,
        "test_fraction": 0.12,
        "per_class": 5000,
        "strict_per_class": True,
        "seed": 2,
        "fallback_base_url": DEFAULT_BASE_URL,
    }
    kwargs.update(overrides)
    return llm_official_result(**kwargs)  # type: ignore[arg-type]


def test_official_result_accepts_only_the_closed_groq_setup():
    official, deviations = _closed_llm_call()
    assert official is True
    assert deviations == []
    zero, zero_deviations = _closed_llm_call(shot="zero", few_shot_k=0, seed=0)
    assert zero is True
    assert zero_deviations == []

    official, deviations = _closed_llm_call(dry_run=True)
    assert official is False
    assert any(item.startswith("dry_run") for item in deviations)

    official, deviations = _closed_llm_call(model_id="meta-llama/llama-3.1-8b-instruct")
    assert official is False
    assert any("revogado" in item for item in deviations)

    official, deviations = _closed_llm_call(test_subsample=None)
    assert official is False
    assert any("test_subsample" in item for item in deviations)

    official, deviations = _closed_llm_call(few_shot_k=5)
    assert official is False
    assert any("k=3" in item for item in deviations)

    official, deviations = _closed_llm_call(seed=7, per_class=10000, strict_per_class=False)
    assert official is False
    assert any("5000" in item for item in deviations)
    assert any("sementes" in item for item in deviations)

    leaked = "https://sk-super-secret@api.groq.com/openai/v1?api_key=sk-super-secret"
    official, deviations = _closed_llm_call(base_url=leaked)
    assert official is True
    assert "sk-super-secret" not in " ".join(deviations)
    assert public_base_url(leaked, fallback=DEFAULT_BASE_URL) == "https://api.groq.com/openai/v1"
    assert public_base_url("not a url", fallback=DEFAULT_BASE_URL) == ""


def test_few_shot_defaults_to_the_closed_k(tiny_path: Path, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    default_path = tmp_path / "default.json"
    explicit_path = tmp_path / "explicit.json"
    shared = [
        "--data",
        str(tiny_path),
        "--shot",
        "few",
        "--test-size",
        "3",
        "--per-class",
        "1",
        "--seed",
        "0",
        "--dry-run",
    ]
    assert main([*shared, "--output", str(default_path)]) == 0
    assert main([*shared, "--k", "3", "--output", str(explicit_path)]) == 0
    default_payload = json.loads(default_path.read_text(encoding="utf-8"))
    explicit_payload = json.loads(explicit_path.read_text(encoding="utf-8"))
    assert default_payload["k"] == 3
    assert default_payload["demonstration_ids"] == explicit_payload["demonstration_ids"]
    assert default_payload["official_result"] is False


def test_test_subsample_keeps_the_official_split(tiny_path: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://sk-super-secret@api.groq.com/openai/v1",
    )
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")

    def _boom(*args, **kwargs):
        raise AssertionError("dry-run chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    full_path = tmp_path / "full.json"
    sub_path = tmp_path / "sub.json"
    shared = [
        "--data",
        str(tiny_path),
        "--shot",
        "zero",
        "--test-size",
        "6",
        "--per-class",
        "1",
        "--seed",
        "0",
        "--split-unit",
        "turn",
        "--dry-run",
    ]
    assert main([*shared, "--output", str(full_path)]) == 0
    assert main([*shared, "--test-subsample", "3", "--output", str(sub_path)]) == 0
    full_payload = json.loads(full_path.read_text(encoding="utf-8"))
    sub_payload = json.loads(sub_path.read_text(encoding="utf-8"))
    assert full_payload["split"]["train_ids"] == sub_payload["split"]["train_ids"]
    assert full_payload["split"]["test_ids"] == sub_payload["split"]["test_ids"]
    assert len(full_payload["predictions"]) == full_payload["split"]["n_test"] == 6
    assert full_payload["test_subsample"]["applied"] is False
    assert len(sub_payload["predictions"]) == 3
    assert sub_payload["test_subsample"]["applied"] is True
    assert sub_payload["test_subsample"]["requested"] == 3
    assert sub_payload["test_subsample"]["actual"] == 3
    assert sum(sub_payload["test_subsample"]["class_counts"].values()) == 3
    assert sub_payload["test_subsample"]["official_test_size"] == 6
    assert sub_payload["test_subsample"]["ids"] == [row["id"] for row in sub_payload["predictions"]]
    assert set(sub_payload["test_subsample"]["ids"]) < set(sub_payload["split"]["test_ids"])
    assert sub_payload["official_result"] is False
    assert sub_payload["live"] is False
    assert sub_payload["model"] == "mock"
    assert sub_payload["model_id"] == "openai/gpt-oss-20b"
    assert sub_payload["base_url"] == "https://api.groq.com/openai/v1"
    assert "sk-super-secret" not in sub_path.read_text(encoding="utf-8")
    assert main([*shared, "--test-subsample", "0"]) == 1
    assert main([*shared, "--test-subsample", "99"]) == 1


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
