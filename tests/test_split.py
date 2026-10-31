import random

import pytest

from user_feedback_analysis.data.load import load_jsonl
from user_feedback_analysis.data.split import (
    SPLIT_FRACTION_RNG_ORDER,
    SPLIT_RNG_ORDER,
    STRATIFY_RULE_CONVERSATION,
    SplitConfig,
    conversation_majority,
    fraction_count,
    make_splits,
    take_holdout,
)
from user_feedback_analysis.labels import LABELS, Label
from user_feedback_analysis.records import Example, LoadedCorpus, RawRow


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
    assert manifest.stratified is False
    assert manifest.test_fraction_requested is None
    assert manifest.test_size_requested == 3
    assert manifest.test_class_targets is None
    assert sum(manifest.test_class_counts.values()) == manifest.n_test
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


def test_fraction_count_uses_half_up():
    assert fraction_count(100, 0.12) == 12
    assert fraction_count(10, 0.12) == 1
    assert fraction_count(5, 0.12) == 1
    assert fraction_count(4, 0.12) == 0
    assert fraction_count(20, 0.12) == 2
    assert fraction_count(5, 0.5) == 3
    assert fraction_count(25, 0.2) == 5


def test_split_config_accepts_one_holdout_knob():
    SplitConfig(seed=0, test_fraction=0.12)
    with pytest.raises(ValueError, match="só um"):
        SplitConfig(seed=0, test_size=1, test_fraction=0.12)
    with pytest.raises(ValueError, match="só um"):
        SplitConfig(seed=0)
    with pytest.raises(ValueError, match="test_fraction"):
        SplitConfig(seed=0, test_fraction=1)
    with pytest.raises(ValueError, match="test_fraction"):
        SplitConfig(seed=0, test_fraction=0)


def test_conversation_majority_tie_follows_label_order():
    sat = _example(0, 0, Label.SAT)
    dsat = _example(1, 0, Label.DSAT)
    no_signal = _example(2, 0, Label.NO_SIGNAL)
    assert conversation_majority([sat, dsat]) is Label.SAT
    assert conversation_majority([dsat, no_signal, dsat, no_signal]) is Label.DSAT
    assert conversation_majority([sat, dsat, no_signal] * 2) is Label.SAT
    assert conversation_majority([no_signal, no_signal, sat]) is Label.NO_SIGNAL


def test_turn_fraction_is_stratified_and_not_rebalanced():
    rows: list[RawRow] = []
    index = 0
    for count, label in (
        (50, Label.SAT),
        (20, Label.DSAT),
        (200, Label.NO_SIGNAL),
    ):
        for _ in range(count):
            rows.append(_raw(index, index, label))
            index += 1
    corpus = LoadedCorpus(rows=rows, conversation_note="teste", source="mem")
    config = SplitConfig(seed=1, test_fraction=0.12, per_class=3, split_unit="turn")
    split = make_splits(corpus, config)
    manifest = split.manifest
    assert manifest.stratified is True
    assert manifest.test_fraction_requested == 0.12
    assert manifest.test_size_requested is None
    assert manifest.seed == 1
    assert manifest.rng_order == SPLIT_FRACTION_RNG_ORDER
    assert manifest.test_class_counts == {"SAT": 6, "DSAT": 2, "sem sinal": 24}
    assert manifest.test_class_targets == manifest.test_class_counts
    assert manifest.eligible_class_counts == {"SAT": 50, "DSAT": 20, "sem sinal": 200}
    assert manifest.test_target_size == 32
    assert manifest.n_test == 32
    assert len(set(manifest.test_class_counts.values())) > 1
    assert manifest.no_signal_pool_before_sample == 176
    assert manifest.no_signal_sampled == 3
    test_ids = {item.example_id for item in split.test}
    others = {item.example_id for item in split.train + split.val + split.unused}
    assert test_ids.isdisjoint(others)
    assert len(test_ids) + len(others) == manifest.n_eligible
    again = make_splits(corpus, config)
    assert [item.example_id for item in again.test] == [item.example_id for item in split.test]
    other = make_splits(
        corpus,
        SplitConfig(seed=2, test_fraction=0.12, per_class=3, split_unit="turn"),
    )
    assert other.manifest.test_class_counts == manifest.test_class_counts
    assert [item.example_id for item in other.test] != [item.example_id for item in split.test]


def test_conversation_fraction_keeps_groups_and_can_overshoot():
    rows: list[RawRow] = []
    index = 0

    def add(conversation_id: int, label: Label, count: int) -> None:
        nonlocal index
        for _ in range(count):
            rows.append(_raw(index, conversation_id, label))
            index += 1

    add(0, Label.NO_SIGNAL, 10)
    add(1, Label.NO_SIGNAL, 10)
    for conversation_id in range(2, 12):
        add(conversation_id, Label.SAT, 1)
    for conversation_id in range(12, 22):
        add(conversation_id, Label.DSAT, 1)
    add(22, Label.NO_SIGNAL, 5)
    corpus = LoadedCorpus(rows=rows, conversation_note="teste", source="mem")
    split = make_splits(
        corpus,
        SplitConfig(
            seed=0,
            test_fraction=0.2,
            per_class=1,
            val_size=1,
            split_unit="conversation",
        ),
        rng=IdentityRng(),
    )
    manifest = split.manifest
    assert manifest.stratify_rule == STRATIFY_RULE_CONVERSATION
    assert manifest.test_class_targets == {"SAT": 2, "DSAT": 2, "sem sinal": 5}
    assert manifest.test_class_counts == {"SAT": 2, "DSAT": 2, "sem sinal": 10}
    assert manifest.test_target_size == 9
    assert manifest.n_test == 14
    assert manifest.n_test > manifest.test_target_size
    _assert_conversation_holdout(split)
    assert {item.conversation_id for item in split.test} == {0, 2, 3, 12, 13}
    assert {item.conversation_id for item in split.val} == {1}
    assert manifest.no_signal_sampled == 1
    assert manifest.no_signal_pool_before_sample == 5


def test_tied_conversation_is_not_split_across_the_holdout():
    rows = [
        _raw(0, 0, Label.SAT),
        _raw(1, 0, Label.DSAT),
        _raw(2, 1, Label.SAT),
        _raw(3, 2, Label.SAT),
        _raw(4, 3, Label.SAT),
        _raw(5, 4, Label.DSAT),
        _raw(6, 5, Label.DSAT),
        _raw(7, 6, Label.DSAT),
        _raw(8, 7, Label.DSAT),
        _raw(9, 8, Label.NO_SIGNAL),
        _raw(10, 9, Label.NO_SIGNAL),
        _raw(11, 10, Label.NO_SIGNAL),
        _raw(12, 11, Label.NO_SIGNAL),
    ]
    corpus = LoadedCorpus(rows=rows, conversation_note="teste", source="mem")
    split = make_splits(
        corpus,
        SplitConfig(seed=0, test_fraction=0.5, per_class=1, split_unit="conversation"),
        rng=IdentityRng(),
    )
    assert conversation_majority(rows_as_examples(split, "r0", "r1")) is Label.SAT
    test_ids = {item.example_id for item in split.test}
    assert {"r0", "r1"} <= test_ids
    _assert_conversation_holdout(split)
    assert split.manifest.test_class_targets == {"SAT": 2, "DSAT": 3, "sem sinal": 2}


def test_tiny_fraction_refuses_an_empty_test():
    rows = [_raw(index, index, label) for index, label in enumerate(LABELS)]
    corpus = LoadedCorpus(rows=rows, conversation_note="teste", source="mem")
    with pytest.raises(ValueError, match="nenhum exemplo"):
        make_splits(
            corpus,
            SplitConfig(seed=0, test_fraction=0.12, per_class=1, split_unit="turn"),
        )


def _assert_conversation_holdout(split) -> None:
    test_conversations = {item.conversation_id for item in split.test}
    val_conversations = {item.conversation_id for item in split.val}
    train_side = {item.conversation_id for item in split.train + split.unused}
    assert test_conversations.isdisjoint(val_conversations)
    assert test_conversations.isdisjoint(train_side)
    assert val_conversations.isdisjoint(train_side)


def rows_as_examples(split, *example_ids: str) -> list[Example]:
    by_id = {item.example_id: item for item in split.train + split.val + split.test + split.unused}
    return [by_id[example_id] for example_id in example_ids]


def _raw(index: int, conversation_id: int, label: Label) -> RawRow:
    return RawRow(
        source_index=index,
        conversation_id=conversation_id,
        turn_id=None,
        utterance_id=index,
        role="User",
        content=f"row {index}",
        satisfaction=label is Label.SAT,
        dissatisfaction=label is Label.DSAT,
    )


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
