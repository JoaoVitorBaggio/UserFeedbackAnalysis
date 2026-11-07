"""Subamostra estratificada do teste oficial. Sem rede."""

import random

import pytest

from user_feedback_analysis.data.load import corpus_from_mappings
from user_feedback_analysis.data.split import SplitConfig, make_splits
from user_feedback_analysis.data.subsample import (
    subsample_class_targets,
    subsample_stratified,
)
from user_feedback_analysis.labels import LABELS, Label
from user_feedback_analysis.records import Example


def _example(index: int, label: Label) -> Example:
    return Example(
        example_id=f"r{index}",
        conversation_id=index,
        text=f"text-{index}",
        label=label,
        role="User",
        source_index=index,
    )


def _pool(counts: dict[Label, int]) -> list[Example]:
    examples: list[Example] = []
    index = 0
    for label in LABELS:
        for _ in range(counts[label]):
            examples.append(_example(index, label))
            index += 1
    return examples


def test_targets_follow_largest_remainder_and_label_ties():
    assert subsample_class_targets({"SAT": 2, "DSAT": 2, "sem sinal": 2}, 2) == {
        "SAT": 1,
        "DSAT": 1,
        "sem sinal": 0,
    }
    assert subsample_class_targets({"SAT": 2, "DSAT": 2, "sem sinal": 2}, 1) == {
        "SAT": 1,
        "DSAT": 0,
        "sem sinal": 0,
    }
    assert subsample_class_targets({"SAT": 6, "DSAT": 2, "sem sinal": 24}, 10) == {
        "SAT": 2,
        "DSAT": 1,
        "sem sinal": 7,
    }


def test_targets_for_the_published_seed0_mix_sum_to_1000():
    # Mistura do teste da semente 0 em docs/cardiff-baseline-2026-11-02.md.
    counts = {"SAT": 1445, "DSAT": 4259, "sem sinal": 57296}
    targets = subsample_class_targets(counts, 1000)
    assert targets == {"SAT": 23, "DSAT": 68, "sem sinal": 909}
    assert sum(targets.values()) == 1000
    assert all(targets[name] <= counts[name] for name in counts)


def test_subsample_is_deterministic_stratified_and_does_not_mutate():
    examples = _pool({Label.SAT: 30, Label.DSAT: 20, Label.NO_SIGNAL: 80})
    order = [item.example_id for item in examples]
    first = subsample_stratified(examples, 20, seed=0)
    again = subsample_stratified(examples, 20, seed=0)
    other = subsample_stratified(examples, 20, seed=1)
    assert [item.example_id for item in examples] == order
    assert [item.example_id for item in first.examples] == [
        item.example_id for item in again.examples
    ]
    assert [item.example_id for item in first.examples] != [
        item.example_id for item in other.examples
    ]
    assert first.actual == 20
    assert first.requested == 20
    assert sum(first.class_counts.values()) == 20
    assert first.class_counts == first.class_targets
    assert first.class_counts["sem sinal"] > first.class_counts["SAT"] > 0
    assert first.class_counts["DSAT"] > 0
    selected = {item.example_id for item in first.examples}
    assert selected <= {item.example_id for item in examples}
    assert [item.example_id for item in first.examples] == [
        item.example_id for item in examples if item.example_id in selected
    ]


def test_full_request_keeps_every_example_in_order():
    examples = _pool({Label.SAT: 2, Label.DSAT: 2, Label.NO_SIGNAL: 3})
    result = subsample_stratified(examples, len(examples), seed=4)
    assert [item.example_id for item in result.examples] == [
        item.example_id for item in examples
    ]
    assert result.source_class_counts == {"SAT": 2, "DSAT": 2, "sem sinal": 3}


def test_subsample_rejects_bad_sizes_and_unknown_labels():
    examples = _pool({Label.SAT: 1, Label.DSAT: 1, Label.NO_SIGNAL: 1})
    with pytest.raises(ValueError, match="inteiro >= 1"):
        subsample_stratified(examples, 0, seed=0)
    with pytest.raises(ValueError, match="inteiro >= 1"):
        subsample_class_targets({"SAT": 1, "DSAT": 1, "sem sinal": 1}, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="passa do teste"):
        subsample_stratified(examples, 4, seed=0)
    with pytest.raises(ValueError, match="semente"):
        subsample_stratified(examples, 1, seed=-1)
    with pytest.raises(ValueError, match="semente"):
        subsample_stratified(examples, 1, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Não há teste"):
        subsample_stratified([], 1, seed=0)
    with pytest.raises(ValueError, match="fora do protocolo"):
        subsample_class_targets({"SAT": 1, "outro": 1}, 1)


def test_shuffle_ignores_n_so_a_larger_draw_can_add_seats():
    examples = _pool({Label.SAT: 10, Label.DSAT: 10, Label.NO_SIGNAL: 10})
    small_targets = subsample_class_targets(
        {"SAT": 10, "DSAT": 10, "sem sinal": 10}, 6
    )
    large_targets = subsample_class_targets(
        {"SAT": 10, "DSAT": 10, "sem sinal": 10}, 9
    )
    assert all(small_targets[name] <= large_targets[name] for name in small_targets)
    small = subsample_stratified(examples, 6, seed=2)
    large = subsample_stratified(examples, 9, seed=2)
    assert {item.example_id for item in small.examples} <= {
        item.example_id for item in large.examples
    }


def test_subsample_runs_after_the_official_split_and_leaves_it_alone():
    rows = []
    index = 0
    for count, label in ((50, Label.SAT), (20, Label.DSAT), (200, Label.NO_SIGNAL)):
        for _ in range(count):
            rows.append(
                {
                    "UtterranceId": 0,
                    "Role": "User",
                    "Content": f"row {index}",
                    "Satisfaction": label is Label.SAT,
                    "Disatisfaction": label is Label.DSAT,
                }
            )
            index += 1
    corpus = corpus_from_mappings(rows, source="mem")
    config = SplitConfig(seed=1, test_fraction=0.12, per_class=3, split_unit="turn")
    split = make_splits(corpus, config)
    train_ids = [item.example_id for item in split.train]
    test_ids = [item.example_id for item in split.test]
    result = subsample_stratified(split.test, 10, seed=config.seed)
    assert [item.example_id for item in split.train] == train_ids
    assert [item.example_id for item in split.test] == test_ids
    assert result.source_size == split.manifest.n_test == 32
    assert result.source_class_counts == split.manifest.test_class_counts
    assert result.class_counts == {"SAT": 2, "DSAT": 1, "sem sinal": 7}
    assert set(result.to_dict()["ids"]) < set(test_ids)
    assert set(result.to_dict()["ids"]).isdisjoint(train_ids)
    again = make_splits(corpus, config)
    assert [item.example_id for item in again.train] == train_ids
    assert [item.example_id for item in again.test] == test_ids


def test_rng_stream_is_independent_of_the_split_rng():
    examples = _pool({Label.SAT: 8, Label.DSAT: 8, Label.NO_SIGNAL: 8})
    seeded = subsample_stratified(examples, 6, seed=0)
    draw = random.Random(0)
    draw.random()
    # O helper não aceita RNG injetado: a semente recomeça o fluxo.
    assert [item.example_id for item in seeded.examples] == [
        item.example_id for item in subsample_stratified(list(examples), 6, seed=0).examples
    ]
