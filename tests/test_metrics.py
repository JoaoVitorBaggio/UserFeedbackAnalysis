import json
from pathlib import Path

import pytest

from user_feedback_analysis.harness import format_report, main, read_label_file
from user_feedback_analysis.metrics import evaluate

GOLD = ["SAT", "SAT", "DSAT", "DSAT", "sem sinal", "sem sinal"]
PRED = ["SAT", "DSAT", "DSAT", "SAT", "sem sinal", "SAT"]


def test_known_confusion_matches_hand_count():
    report = evaluate(GOLD, PRED)

    sat = report.per_class["SAT"]
    dsat = report.per_class["DSAT"]
    no_signal = report.per_class["sem sinal"]

    assert (sat.tp, sat.fp, sat.fn) == (1, 2, 1)
    assert sat.precision == pytest.approx(1 / 3)
    assert sat.recall == pytest.approx(1 / 2)
    assert sat.f1 == pytest.approx(0.4)

    assert (dsat.tp, dsat.fp, dsat.fn) == (1, 1, 1)
    assert dsat.precision == pytest.approx(0.5)
    assert dsat.recall == pytest.approx(0.5)
    assert dsat.f1 == pytest.approx(0.5)

    assert (no_signal.tp, no_signal.fp, no_signal.fn) == (1, 0, 1)
    assert no_signal.precision == pytest.approx(1.0)
    assert no_signal.recall == pytest.approx(0.5)
    assert no_signal.f1 == pytest.approx(2 / 3)

    assert report.macro_precision == pytest.approx((1 / 3 + 0.5 + 1.0) / 3)
    assert report.macro_recall == pytest.approx(0.5)
    assert report.macro_f1 == pytest.approx((0.4 + 0.5 + 2 / 3) / 3)
    assert report.macro_f1 == pytest.approx(47 / 90)
    assert report.n == 6
    assert report.confusion == (
        (1, 1, 0),
        (1, 1, 0),
        (1, 0, 1),
    )


def test_absent_class_uses_zero_division():
    report = evaluate(["no-signal"], ["sem sinal"])
    assert report.per_class["sem sinal"].f1 == pytest.approx(1.0)
    assert report.per_class["SAT"].support == 0
    assert report.per_class["SAT"].predicted == 0
    assert report.per_class["SAT"].precision == 0.0
    assert report.per_class["SAT"].recall == 0.0
    assert report.per_class["SAT"].f1 == 0.0
    assert report.per_class["DSAT"].f1 == 0.0
    assert report.macro_f1 == pytest.approx(1 / 3)
    assert report.zero_division == 0


def test_perfect_predictions_score_one():
    labels = ["SAT", "DSAT", "sem sinal"]
    report = evaluate(labels, labels)
    assert report.macro_precision == pytest.approx(1.0)
    assert report.macro_recall == pytest.approx(1.0)
    assert report.macro_f1 == pytest.approx(1.0)


def test_length_mismatch_and_unknown_label_fail():
    with pytest.raises(ValueError, match="tamanhos"):
        evaluate(["SAT"], ["SAT", "DSAT"])
    with pytest.raises(ValueError, match="fora"):
        evaluate(["SAT"], ["talvez"])


def test_format_report_names_the_comparison_metric():
    text = format_report(evaluate(GOLD, PRED))
    assert "sem sinal" in text
    assert "macro" in text
    assert "macro F1" in text


def test_harness_cli_on_the_known_confusion(tmp_path: Path):
    gold = tmp_path / "gold.txt"
    pred = tmp_path / "pred.txt"
    output = tmp_path / "report.json"
    gold.write_text("\n".join(GOLD) + "\n", encoding="utf-8")
    pred.write_text("\n".join(PRED) + "\n", encoding="utf-8")
    assert main(["--gold", str(gold), "--pred", str(pred), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["official_result"] is False
    assert payload["metrics"]["macro_f1"] == pytest.approx(47 / 90)
    assert read_label_file(gold) == GOLD
