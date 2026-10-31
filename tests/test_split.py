import random

import pytest

from user_feedback_analysis.data.load import load_jsonl
from user_feedback_analysis.data.split import (
    SPLIT_RNG_ORDER,
    SplitConfig,
    make_splits,
    take_holdout,
)
from user_feedback_analysis.labels import Label
from user_feedback_analysis.records import Example


class IdentityRng(random.Random):
    def shuffle(self, seq):
        return None

    def sample(self, population, k):
        return list(population)[:k]


def _example(index: int, conversation_id: int, label: Label) -> Example:
    return Example(
        example_id=f"r{index}",
        conversation_id=conversation_id,
        text=f"text-{index}",
        label=label,
        role="User",
        source_index=index,
    )


def test_turn_split_can_share_a_conversation():
    examples = [
        _example(0, 0, Label.SAT),
        _example(1, 0, Label.DSAT),
        _example(2, 1, Label.NO_SIGNAL),
    ]
    holdout, rest = take_holdout(examples, 1, IdentityRng(), "turn")
    assert {item.conversation_id for item in holdout} & {item.conversation_id for item in rest}


def test_conversation_holdout_keeps_the_group_and_can_overshoot():
    examples = [
        _example(0, 0, Label.SAT),
        _example(1, 0, Label.DSAT),
        _example(2, 1, Label.NO_SIGNAL),
        _example(3, 2, Label.SAT),
    ]
    holdout, rest = take_holdout(examples, 1, IdentityRng(), "conversation")
    assert [item.example_id for item in holdout] == ["r0", "r1"]
    assert {item.conversation_id for item in holdout}.isdisjoint(
        {item.conversation_id for item in rest}
    )


def test_fixture_filters_and_shared_conversation(tiny_path):
    corpus = load_jsonl(tiny_path)
    split = make_splits(
        corpus,
        SplitConfig(seed=0, test_size=3, per_class=1, split_unit="conversation"),
    )
    manifest = split.manifest
    assert manifest.n_rows == 18
    assert manifest.n_dropped_role == 3
    assert manifest.n_dropped_ambiguous == 1
    assert manifest.n_dropped_empty == 1
    assert manifest.n_eligible == 13
    assert manifest.rng_order == SPLIT_RNG_ORDER
    assert manifest.no_signal_sampled == 1
    assert manifest.no_signal_pool_before_sample > manifest.no_signal_sampled

    parts = split.train + split.val + split.test + split.unused
    assert len(parts) == manifest.n_eligible
    assert len({item.example_id for item in parts}) == manifest.n_eligible
    assert all(item.role.casefold() == "user" for item in parts)
    assert all(item.label is not None for item in parts)

    by_id = {item.example_id: item for item in parts}
    assert by_id["r16"].label is Label.SAT
    assert by_id["r14"].label is Label.NO_SIGNAL
    assert by_id["r14"].conversation_id == by_id["r16"].conversation_id
    assert by_id["r17"].label is Label.DSAT

    located = {}
    for name, group in (
        ("test", split.test),
        ("train", split.train),
        ("unused", split.unused),
    ):
        for item in group:
            located[item.example_id] = name
    assert located["r14"] == located["r16"]
    test_conversations = {item.conversation_id for item in split.test}
    other_conversations = {
        item.conversation_id for item in split.train + split.val + split.unused
    }
    assert test_conversations.isdisjoint(other_conversations)

    again = make_splits(
        corpus,
        SplitConfig(seed=0, test_size=3, per_class=1),
    )
    assert [item.example_id for item in again.test] == [item.example_id for item in split.test]
    assert [item.example_id for item in again.train] == [item.example_id for item in split.train]


def test_no_signal_is_a_sample_and_test_comes_first():
    rows = []
    index = 0
    for _ in range(2):
        rows.append(_mapping(index, Label.SAT))
        index += 1
    for _ in range(2):
        rows.append(_mapping(index, Label.DSAT))
        index += 1
    for _ in range(6):
        rows.append(_mapping(index, Label.NO_SIGNAL))
        index += 1
    from user_feedback_analysis.data.load import corpus_from_mappings

    corpus = corpus_from_mappings(rows, source="mem")
    split = make_splits(
        corpus,
        SplitConfig(seed=1, test_size=1, per_class=1, split_unit="turn"),
        rng=IdentityRng(),
    )
    assert split.manifest.no_signal_sampled == 1
    assert split.manifest.no_signal_pool_before_sample > 1
    test_ids = {item.example_id for item in split.test}
    assert test_ids.isdisjoint({item.example_id for item in split.train})
    no_signal_train = [item for item in split.train if item.label is Label.NO_SIGNAL]
    assert len(no_signal_train) == 1
    assert no_signal_train[0].example_id not in test_ids


def test_strict_per_class_refuses_a_short_pool():
    from user_feedback_analysis.data.load import corpus_from_mappings

    rows = []
    index = 0
    for label in (Label.SAT, Label.DSAT, Label.NO_SIGNAL):
        for _ in range(2):
            rows.append(_mapping(index, label))
            index += 1
    corpus = corpus_from_mappings(rows, source="mem")
    config = SplitConfig(
        seed=0,
        test_size=1,
        per_class=2,
        strict_per_class=True,
        split_unit="turn",
    )
    with pytest.raises(ValueError, match="strict_per_class"):
        make_splits(corpus, config, rng=IdentityRng())


def test_validation_stays_off_the_test_set():
    from user_feedback_analysis.data.load import corpus_from_mappings

    rows = []
    index = 0
    for label in (Label.SAT, Label.DSAT, Label.NO_SIGNAL):
        for _ in range(4):
            rows.append(_mapping(index, label))
            index += 1
    corpus = corpus_from_mappings(rows, source="mem")
    split = make_splits(
        corpus,
        SplitConfig(seed=0, test_size=1, val_size=1, per_class=1, split_unit="conversation"),
        rng=IdentityRng(),
    )
    test_ids = {item.example_id for item in split.test}
    val_ids = {item.example_id for item in split.val}
    train_ids = {item.example_id for item in split.train}
    assert split.manifest.n_val >= 1
    assert test_ids.isdisjoint(val_ids)
    assert test_ids.isdisjoint(train_ids)
    assert val_ids.isdisjoint(train_ids)
    assert {item.conversation_id for item in split.test}.isdisjoint(
        {item.conversation_id for item in split.val + split.train}
    )


def test_split_config_rejects_bool_seed():
    with pytest.raises(ValueError):
        SplitConfig(seed=True, test_size=1)  # type: ignore[arg-type]


def _mapping(index: int, label: Label) -> dict[str, object]:
    satisfaction = label is Label.SAT
    dissatisfaction = label is Label.DSAT
    return {
        "UtterranceId": 0,
        "Role": "User",
        "Content": f"row {index}",
        "Satisfaction": satisfaction,
        "Disatisfaction": dissatisfaction,
    }
