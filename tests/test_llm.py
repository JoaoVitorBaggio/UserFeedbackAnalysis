import http.client
import io
import json
import os
import ssl
import subprocess
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
    RefusalFallback,
    VoteResult,
    _RETRY_CAP_S,
    chat_complete,
    classify_text,
    classify_votes,
    resolve_dry_run,
)
from user_feedback_analysis.llm.protocol import (
    OFFICIAL_RESULT_POLICY,
    PROTOCOL_LLM_BASE_URL,
    PROTOCOL_LLM_MODEL_ID,
    REFUSAL_FALLBACK_PHRASE,
    REVOKED_LLM_BASE_URL,
    apply_refusal_deviation,
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


# Recusa real do gpt-oss em 2026-11-04, com apóstrofo curvo.
_REFUSAL = "I\u2019m sorry, but I can\u2019t help with that."


def test_parse_rejects_two_labels_and_empty_text():
    with pytest.raises(ValueError):
        parse_label("SAT and DSAT")
    with pytest.raises(ValueError):
        parse_label("hello")
    with pytest.raises(ValueError, match="rótulo único"):
        parse_label(_REFUSAL)
    with pytest.raises(ValueError, match="rótulo único"):
        parse_label("")


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


def _message_response(message: dict):
    encoded = json.dumps({"choices": [{"message": message}]}).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return encoded

    return _Response()


def _ok_response(content: str = "SAT"):
    return _message_response({"content": content})


def _install_urlopen(monkeypatch, outcomes: list[object]) -> dict[str, object]:
    seen: dict[str, object] = {"calls": 0, "agents": [], "auths": [], "bodies": []}

    def _urlopen(request, timeout=0):
        seen["calls"] = int(seen["calls"]) + 1
        seen["agents"].append(request.get_header("User-agent"))
        seen["auths"].append(request.get_header("Authorization"))
        seen["timeout"] = timeout
        raw_body = request.data
        if isinstance(raw_body, bytes):
            seen["bodies"].append(json.loads(raw_body.decode("utf-8")))
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


def _bodies(seen: dict[str, object]) -> list[dict]:
    bodies = seen["bodies"]
    assert isinstance(bodies, list)
    return bodies


def test_chat_complete_retries_a_refusal_then_keeps_the_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [
            _ok_response(_REFUSAL),
            _ok_response("The label is DSAT."),
        ],
    )
    assert chat_complete("Classify this.", model="openai/gpt-oss-20b", temperature=0.7) == (
        "The label is DSAT."
    )
    assert seen["calls"] == 2
    assert sleeps == [1.0]
    bodies = _bodies(seen)
    assert bodies[0] == bodies[1]
    assert bodies[0]["temperature"] == 0.7
    assert bodies[0]["messages"][1]["content"] == "Classify this."
    assert "sk-test" not in json.dumps(bodies)


def test_chat_complete_retries_empty_content_and_ignores_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    hidden = "cadeia secreta SAT"
    seen = _install_urlopen(
        monkeypatch,
        [
            _message_response({"content": None, "reasoning_content": hidden}),
            _message_response({"reasoning_content": "DSAT"}),
            _message_response(
                {"content": "  ", "reasoning_content": hidden},
            ),
            _message_response(
                {"content": "DSAT", "reasoning_content": "SAT and " + _REFUSAL},
            ),
        ],
    )
    assert chat_complete("Classify.", model="m") == "DSAT"
    assert seen["calls"] == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_chat_complete_shares_attempt_budget_between_429_and_refusal(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [
            _http_error(429, "Please try again in 1.5s."),
            _ok_response(_REFUSAL),
            _ok_response("SAT"),
        ],
    )
    assert chat_complete("Classify.", model="m", temperature=0.7) == "SAT"
    assert seen["calls"] == 3
    # A recusa no meio da fila não recomeça o backoff em 1s.
    assert sleeps == [1.5, 2.0]
    assert _bodies(seen)[1] == _bodies(seen)[2]


def test_chat_complete_reads_the_label_from_content_when_reasoning_is_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [
            _message_response(
                {
                    "content": "SAT",
                    "reasoning_content": "I’m unsure. DSAT. " + _REFUSAL,
                }
            )
        ],
    )
    assert chat_complete("Classify.", model="m") == "SAT"
    assert seen["calls"] == 1
    assert sleeps == []


def test_chat_complete_counts_a_persistent_refusal_as_no_signal(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [_ok_response(_REFUSAL) for _ in range(LLM_MAX_ATTEMPTS)],
    )
    refusal = chat_complete("Classify.", model="m")
    assert refusal == "sem sinal"
    assert refusal.refusal_fallback is True
    assert refusal.raw_response == _REFUSAL
    assert seen["calls"] == LLM_MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

    hidden = "cadeia secreta DSAT"
    empty = _install_urlopen(
        monkeypatch,
        [
            _message_response({"content": "", "reasoning_content": hidden})
            for _ in range(LLM_MAX_ATTEMPTS)
        ],
    )
    sleeps.clear()
    blank = chat_complete("Classify.", model="m")
    assert blank == "sem sinal"
    assert blank.refusal_fallback is True
    assert blank.raw_response == ""
    assert hidden not in blank
    assert hidden not in blank.raw_response
    assert "DSAT" not in blank.raw_response
    assert empty["calls"] == LLM_MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def test_exhausted_429_still_raises_when_the_last_attempt_is_not_a_refusal(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [
            *[_ok_response(_REFUSAL) for _ in range(LLM_MAX_ATTEMPTS - 1)],
            _http_error(429, "Please try again in 1s."),
        ],
    )
    with pytest.raises(RuntimeError, match=rf"LLM HTTP 429:.*após {LLM_MAX_ATTEMPTS} tentativas"):
        chat_complete("Classify.", model="m")
    assert seen["calls"] == LLM_MAX_ATTEMPTS
    # A 429 da última tentativa não vira sem sinal e não dorme de novo.
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def _raw_response(payload: dict):
    encoded = json.dumps(payload).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return encoded

    return _Response()


def test_chat_complete_does_not_retry_a_broken_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)

    listed = _install_urlopen(
        monkeypatch,
        [_message_response({"content": ["SAT"]})],
    )
    with pytest.raises(RuntimeError, match="não é texto"):
        chat_complete("Classify.", model="m")
    assert listed["calls"] == 1
    assert sleeps == []

    broken = _install_urlopen(monkeypatch, [_raw_response({"nope": True})])
    with pytest.raises(RuntimeError, match="sem choices"):
        chat_complete("Classify.", model="m")
    assert broken["calls"] == 1

    garbled = _install_urlopen(monkeypatch, [_raw_response_bytes(b"not-json")])
    with pytest.raises(RuntimeError, match="sem choices"):
        chat_complete("Classify.", model="m")
    assert garbled["calls"] == 1
    assert sleeps == []


def _raw_response_bytes(payload: bytes):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    return _Response()


def test_classify_votes_retries_one_inference_inside_chat_complete(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [
            _ok_response(_REFUSAL),
            _ok_response("SAT"),
            _message_response({"content": None, "reasoning_content": "SAT"}),
            _ok_response("DSAT"),
            _ok_response("The label is DSAT."),
        ],
    )
    voted = classify_votes(
        "List three colors.",
        [],
        dry_run=False,
        model="dummy-model",
        max_chars=100,
        k=3,
        temperature=0.4,
    )
    assert voted.votes == (Label.SAT, Label.DSAT, Label.DSAT)
    assert voted.label is Label.DSAT
    assert voted.decision == "majority"
    assert voted.refusals == ()
    assert seen["calls"] == 5
    assert sleeps == [1.0, 1.0]
    bodies = _bodies(seen)
    assert all(body["temperature"] == 0.4 for body in bodies)
    assert bodies[0] == bodies[1]
    assert bodies[0]["messages"][1]["content"] == bodies[3]["messages"][1]["content"]


def test_injected_complete_is_not_wrapped_in_a_second_retry(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    calls = {"n": 0}

    def complete(prompt: str, *, model: str, timeout: float, temperature: float) -> str:
        calls["n"] += 1
        return _REFUSAL

    with pytest.raises(ValueError, match="rótulo único"):
        classify_votes(
            "List three colors.",
            [],
            dry_run=False,
            model="dummy-model",
            max_chars=None,
            k=1,
            complete=complete,
        )
    assert calls["n"] == 1
    assert sleeps == []


def test_classify_votes_keeps_majority_when_one_inference_falls_back(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    seen = _install_urlopen(
        monkeypatch,
        [
            *[_ok_response(_REFUSAL) for _ in range(LLM_MAX_ATTEMPTS)],
            _ok_response("SAT"),
            _ok_response("The label is SAT."),
        ],
    )
    voted = classify_votes(
        "List three colors.",
        [],
        dry_run=False,
        model="dummy-model",
        max_chars=100,
        k=3,
        temperature=0.4,
    )
    assert voted.votes == (Label.NO_SIGNAL, Label.SAT, Label.SAT)
    assert voted.label is Label.SAT
    assert voted.decision == "majority"
    assert voted.refusals == (RefusalFallback(index=0, raw_response=_REFUSAL),)
    assert seen["calls"] == LLM_MAX_ATTEMPTS + 2
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def test_persistent_refusal_does_not_stop_the_seed(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", lambda _seconds: None)
    output = tmp_path / "live.json"
    args = [
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
        "--inferences",
        "1",
        "--live",
        "--output",
        str(output),
    ]
    seen = _install_urlopen(
        monkeypatch,
        [
            _ok_response("SAT"),
            *[_ok_response(_REFUSAL) for _ in range(LLM_MAX_ATTEMPTS)],
            _ok_response("DSAT"),
        ],
    )
    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "kind" not in payload
    assert seen["calls"] == 2 + LLM_MAX_ATTEMPTS
    assert payload["official_result"] is False
    assert isinstance(payload["metrics"]["macro_f1"], float)
    assert payload["predictions"][0]["id"] == "r2"
    assert payload["predictions"][0]["pred"] == "SAT"
    assert "refusal_fallback" not in payload["predictions"][0]
    refused = payload["predictions"][1]
    assert refused["id"] == "r13"
    assert refused["pred"] == "sem sinal"
    assert refused["refusal_fallback"] is True
    assert refused["refusal_responses"] == [{"index": 0, "text": _REFUSAL}]
    assert payload["inference_votes"][1] == ["sem sinal"]
    assert payload["predictions"][2]["pred"] == "DSAT"
    blob = " ".join(payload["protocol_deviations"])
    assert f"{REFUSAL_FALLBACK_PHRASE}: 1 inferência (r13[0])." in payload["protocol_deviations"]
    assert "r13[0]" in blob
    assert "sk-test" not in output.read_text(encoding="utf-8")
    assert "Recusa persistente" in payload["note"]


def test_refusal_is_recorded_in_the_checkpoint_and_kept_on_resume(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai")
    sleeps: list[float] = []
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", sleeps.append)
    output = tmp_path / "live.json"
    args = [
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
        "--inferences",
        "1",
        "--live",
        "--output",
        str(output),
    ]
    first = _install_urlopen(
        monkeypatch,
        [
            _ok_response("SAT"),
            *[_ok_response(_REFUSAL) for _ in range(LLM_MAX_ATTEMPTS)],
            *[_http_error(429, "Please try again in 1s.") for _ in range(LLM_MAX_ATTEMPTS)],
        ],
    )
    assert main(args) == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["kind"] == "llm_checkpoint"
    assert saved["version"] == 2
    assert saved["fingerprint"]["version"] == 1
    assert saved["official_result"] is False
    assert saved["next_index"] == 2
    assert saved["n_completed"] == 2
    assert saved["completed"][0]["pred"] == "SAT"
    assert saved["completed"][0]["votes"] == ["SAT"]
    assert "refusal_fallback" not in saved["completed"][0]
    refused = saved["completed"][1]
    assert refused["id"] == "r13"
    assert refused["pred"] == "sem sinal"
    assert refused["votes"] == ["sem sinal"]
    assert refused["refusal_fallback"] is True
    assert refused["refusal_responses"] == [{"index": 0, "text": _REFUSAL}]
    assert "sk-test" not in output.read_text(encoding="utf-8")
    assert first["calls"] == 1 + (2 * LLM_MAX_ATTEMPTS)
    assert sleeps[:7] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    refusal_bodies = _bodies(first)[1:1 + LLM_MAX_ATTEMPTS]
    assert len(refusal_bodies) == LLM_MAX_ATTEMPTS
    assert all(body == refusal_bodies[0] for body in refusal_bodies)
    assert refusal_bodies[0]["temperature"] == 0.7

    second = _install_urlopen(monkeypatch, [_ok_response("DSAT")])
    sleeps.clear()
    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "kind" not in payload
    assert payload["resume"]["n_reused"] == 2
    assert payload["official_result"] is False
    assert payload["dry_run"] is False
    assert payload["predictions"][0]["pred"] == "SAT"
    assert payload["predictions"][1]["pred"] == "sem sinal"
    assert payload["predictions"][1]["refusal_fallback"] is True
    assert payload["predictions"][1]["refusal_responses"] == [{"index": 0, "text": _REFUSAL}]
    assert payload["predictions"][2]["pred"] == "DSAT"
    assert "refusal_fallback" not in payload["predictions"][2]
    assert second["calls"] == 1
    assert sleeps == []
    blob = " ".join(payload["protocol_deviations"])
    assert f"{REFUSAL_FALLBACK_PHRASE}: 1 inferência (r13[0])." in payload["protocol_deviations"]
    assert "checkpoint" not in blob
    assert "content vazio" not in blob
    assert "sk-test" not in output.read_text(encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise AssertionError("retomada do JSON final refez inferência")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert main(args) == 0
    again = json.loads(output.read_text(encoding="utf-8"))
    assert again["resume"]["n_reused"] == 3
    assert again["predictions"] == payload["predictions"]
    assert again["protocol_deviations"] == payload["protocol_deviations"]
    assert again["official_result"] is False
    assert again["inference_votes"] == payload["inference_votes"]


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
    assert payload["model_requirement"]["accepted_backend"] == "DeepInfra"
    assert payload["model_requirement"]["accepted_base_url"] == "https://api.deepinfra.com/v1/openai"
    assert payload["model_requirement"]["backend_decision"] == "2026-11-04"
    assert payload["model_requirement"]["revoked_status"] == "revoked"
    assert payload["model_requirement"]["revoked_base_url"] == "https://api.groq.com/openai/v1"
    assert payload["model_requirement"]["revoked_base_status"] == "revoked"
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
        "base_url": "https://api.deepinfra.com/v1/openai/",
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


def test_official_result_accepts_deepinfra_and_rejects_groq():
    assert PROTOCOL_LLM_BASE_URL == "https://api.deepinfra.com/v1/openai"
    assert REVOKED_LLM_BASE_URL == "https://api.groq.com/openai/v1"
    assert "https://api.deepinfra.com/v1/openai" in OFFICIAL_RESULT_POLICY
    assert "revogada" in OFFICIAL_RESULT_POLICY
    assert REVOKED_LLM_BASE_URL in OFFICIAL_RESULT_POLICY
    assert REFUSAL_FALLBACK_PHRASE in OFFICIAL_RESULT_POLICY
    assert "official_result fica false" in OFFICIAL_RESULT_POLICY

    official, deviations = _closed_llm_call()
    assert official is True
    assert deviations == []
    zero, zero_deviations = _closed_llm_call(shot="zero", few_shot_k=0, seed=0)
    assert zero is True
    assert zero_deviations == []
    for seed in (0, 1, 2):
        accepted, accepted_deviations = _closed_llm_call(seed=seed)
        assert accepted is True
        assert accepted_deviations == []

    official, deviations = _closed_llm_call(dry_run=True)
    assert official is False
    assert any(item.startswith("dry_run") for item in deviations)

    official, deviations = _closed_llm_call(model_id="meta-llama/llama-3.1-8b-instruct")
    assert official is False
    assert any("revogado" in item for item in deviations)
    assert not any(item.startswith("base_url") for item in deviations)

    official, deviations = _closed_llm_call(base_url=REVOKED_LLM_BASE_URL)
    assert official is False
    assert any("revogada" in item and REVOKED_LLM_BASE_URL in item for item in deviations)
    assert any(PROTOCOL_LLM_BASE_URL in item for item in deviations)

    official, deviations = _closed_llm_call(base_url="https://openrouter.ai/api/v1")
    assert official is False
    assert any("openrouter.ai" in item for item in deviations)
    assert any("não é" in item for item in deviations)

    official, deviations = _closed_llm_call(base_url=None)
    assert official is False
    assert any("api.openai.com" in item for item in deviations)

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

    leaked = (
        "https://sk-super-secret@api.deepinfra.com/v1/openai"
        "?api_key=sk-super-secret"
    )
    official, deviations = _closed_llm_call(base_url=leaked)
    assert official is True
    assert deviations == []
    assert "sk-super-secret" not in " ".join(deviations)
    assert public_base_url(leaked, fallback=DEFAULT_BASE_URL) == PROTOCOL_LLM_BASE_URL

    groq_leaked = (
        "https://sk-super-secret@api.groq.com/openai/v1?api_key=sk-super-secret"
    )
    official, deviations = _closed_llm_call(base_url=groq_leaked)
    assert official is False
    assert "sk-super-secret" not in " ".join(deviations)
    assert public_base_url(groq_leaked, fallback=DEFAULT_BASE_URL) == REVOKED_LLM_BASE_URL
    assert any("revogada" in item for item in deviations)
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
        "https://sk-super-secret@api.deepinfra.com/v1/openai",
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
    assert sub_payload["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert not any(item.startswith("base_url") for item in sub_payload["protocol_deviations"])
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


def _write_protocol_corpus(path: Path, per_class: int = 9000) -> None:
    """Um turno de usuário por conversa, o bastante para 12%, 5000 e subamostra 1000."""
    flags = (
        ("SAT", True, False),
        ("DSAT", False, True),
        ("sem sinal", False, False),
    )
    lines: list[str] = []
    for label, satisfaction, dissatisfaction in flags:
        for index in range(per_class):
            lines.append(
                json.dumps(
                    {
                        "TurnId": 1,
                        "UtterranceId": 0,
                        "Role": "User",
                        "Content": f"{label} {index}",
                        "Satisfaction": satisfaction,
                        "Disatisfaction": dissatisfaction,
                        "State": "FEEDBACK",
                    },
                    ensure_ascii=False,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_offline_votes(monkeypatch) -> None:
    def _votes(*args, **kwargs):
        return VoteResult(
            label=Label.SAT,
            votes=(Label.SAT, Label.SAT, Label.SAT),
            decision="majority",
        )

    monkeypatch.setattr("user_feedback_analysis.llm.run.classify_votes", _votes)

    def _boom(*args, **kwargs):
        raise AssertionError("a corrida de teste chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)


def test_apply_refusal_deviation_follows_the_official_result_policy():
    clean, clean_deviations = apply_refusal_deviation(True, [], [])
    assert clean is True
    assert clean_deviations == []
    fallen, fallen_deviations = apply_refusal_deviation(True, [], [("r13", (0,))])
    assert fallen is False
    assert fallen_deviations == [f"{REFUSAL_FALLBACK_PHRASE}: 1 inferência (r13[0])."]
    mixed, mixed_deviations = apply_refusal_deviation(
        False,
        ["dry_run: a chamada não foi à rede."],
        [("r13", (0, 2)), ("r40", (1,))],
    )
    assert mixed is False
    assert mixed_deviations[0].startswith("dry_run")
    assert mixed_deviations[-1] == (
        f"{REFUSAL_FALLBACK_PHRASE}: 3 inferências (r13[0, 2], r40[1])."
    )


def test_recovered_refusal_is_not_a_protocol_deviation(
    tiny_path: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setattr("user_feedback_analysis.llm.client.time.sleep", lambda _seconds: None)
    output = tmp_path / "recovered.json"
    seen = _install_urlopen(
        monkeypatch,
        [
            _ok_response(_REFUSAL),
            _ok_response("SAT"),
            _ok_response(_REFUSAL),
            _ok_response("DSAT"),
            _ok_response(_REFUSAL),
            _ok_response("sem sinal"),
        ],
    )
    assert main(
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
            "--inferences",
            "1",
            "--live",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert seen["calls"] == 6
    assert all("refusal_fallback" not in row for row in payload["predictions"])
    assert all(REFUSAL_FALLBACK_PHRASE not in item for item in payload["protocol_deviations"])
    assert payload["official_result"] is False


def test_closed_call_with_a_refusal_stays_in_the_report_and_is_not_official(
    tmp_path: Path, monkeypatch
):
    corpus = tmp_path / "protocol.jsonl"
    _write_protocol_corpus(corpus)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai")

    def _boom(*_args, **_kwargs):
        raise AssertionError("a corrida de teste chamou a rede")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    state = {"n": 0}

    def _votes(*_args, **_kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return VoteResult(
                label=Label.SAT,
                votes=(Label.NO_SIGNAL, Label.SAT, Label.SAT),
                decision="majority",
                refusals=(RefusalFallback(index=0, raw_response=_REFUSAL),),
            )
        return VoteResult(
            label=Label.SAT,
            votes=(Label.SAT, Label.SAT, Label.SAT),
            decision="majority",
        )

    monkeypatch.setattr("user_feedback_analysis.llm.run.classify_votes", _votes)
    output = tmp_path / "few.json"
    args = [
        "--data",
        str(corpus),
        "--shot",
        "few",
        "--k",
        "3",
        "--test-fraction",
        "0.12",
        "--test-subsample",
        "1000",
        "--per-class",
        "5000",
        "--strict-per-class",
        "--inferences",
        "3",
        "--seed",
        "1",
        "--live",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert state["n"] == 1000
    assert payload["official_result"] is False
    assert isinstance(payload["metrics"]["macro_f1"], float)
    assert len(payload["predictions"]) == 1000
    first = payload["predictions"][0]
    assert first["refusal_fallback"] is True
    assert first["refusal_responses"] == [{"index": 0, "text": _REFUSAL}]
    assert first["pred"] == "SAT"
    assert "refusal_fallback" not in payload["predictions"][1]
    assert payload["protocol_deviations"] == [
        f"{REFUSAL_FALLBACK_PHRASE}: 1 inferência ({first['id']}[0])."
    ]
    assert "official_result false" in payload["note"]
    assert "Recusa persistente" in payload["note"]
    assert "sk-super-secret" not in output.read_text(encoding="utf-8")

    def _no_more(*_args, **_kwargs):
        raise AssertionError("retomada refez inferência")

    monkeypatch.setattr("user_feedback_analysis.llm.run.classify_votes", _no_more)
    assert main(args) == 0
    again = json.loads(output.read_text(encoding="utf-8"))
    assert again["resume"]["n_reused"] == 1000
    assert again["official_result"] is False
    assert again["predictions"] == payload["predictions"]
    assert again["protocol_deviations"] == payload["protocol_deviations"]


def test_runner_marks_deepinfra_official_and_groq_revoked(
    tmp_path: Path, monkeypatch
):
    corpus = tmp_path / "protocol.jsonl"
    _write_protocol_corpus(corpus)
    _install_offline_votes(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai")
    output = tmp_path / "few.json"
    code = main(
        [
            "--data",
            str(corpus),
            "--shot",
            "few",
            "--k",
            "3",
            "--test-fraction",
            "0.12",
            "--test-subsample",
            "1000",
            "--per-class",
            "5000",
            "--strict-per-class",
            "--inferences",
            "3",
            "--seed",
            "2",
            "--live",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["official_result"] is True
    assert payload["protocol_deviations"] == []
    assert payload["live"] is True
    assert payload["dry_run"] is False
    assert payload["model"] == "openai/gpt-oss-20b"
    assert payload["model_id"] == "openai/gpt-oss-20b"
    assert payload["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert payload["model_requirement"]["accepted_base_url"] == payload["base_url"]
    assert payload["shot"] == "few"
    assert payload["k"] == 3
    assert payload["inference_k"] == 3
    assert payload["split"]["seed"] == 2
    assert payload["test_subsample"]["applied"] is True
    assert payload["test_subsample"]["requested"] == 1000
    assert payload["test_subsample"]["actual"] == 1000
    assert len(payload["predictions"]) == 1000
    assert "sk-super-secret" not in output.read_text(encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1/")
    groq_output = tmp_path / "groq.json"
    groq_code = main(
        [
            "--data",
            str(corpus),
            "--shot",
            "few",
            "--k",
            "3",
            "--test-fraction",
            "0.12",
            "--test-subsample",
            "1000",
            "--per-class",
            "5000",
            "--strict-per-class",
            "--inferences",
            "3",
            "--seed",
            "0",
            "--live",
            "--output",
            str(groq_output),
        ]
    )
    assert groq_code == 0
    groq_payload = json.loads(groq_output.read_text(encoding="utf-8"))
    assert groq_payload["official_result"] is False
    assert groq_payload["base_url"] == "https://api.groq.com/openai/v1"
    assert any(
        "revogada" in item and "https://api.groq.com/openai/v1" in item
        for item in groq_payload["protocol_deviations"]
    )
    assert "sk-super-secret" not in groq_output.read_text(encoding="utf-8")


def test_shell_guard_accepts_deepinfra_and_rejects_groq(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "run-llm-gptoss20b-sub1k.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "python-calls.txt"
    python = fake_bin / "python"
    python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$FAKE_PYTHON_LOG\"\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    data = repo / "tests" / "fixtures" / "tiny_sat_dsat.jsonl"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_PYTHON_LOG"] = str(log)
    env["OPENAI_API_KEY"] = "sk-not-a-real-key"
    env["LLM_MODEL"] = "openai/gpt-oss-20b"
    env["OPENAI_BASE_URL"] = "https://api.deepinfra.com/v1/openai/"
    accepted = subprocess.run(
        ["bash", str(script), str(data)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 6
    assert all(call.startswith("-m user_feedback_analysis.llm.run ") for call in calls)
    assert "https://api.deepinfra.com/v1/openai" in accepted.stdout
    assert "sk-not-a-real-key" not in accepted.stdout
    assert "sk-not-a-real-key" not in accepted.stderr

    env["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
    rejected = subprocess.run(
        ["bash", str(script), str(data)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1
    assert "revogada" in rejected.stderr
    assert "https://api.groq.com/openai/v1" in rejected.stderr
    assert len(log.read_text(encoding="utf-8").splitlines()) == 6
    assert "sk-not-a-real-key" not in rejected.stderr
