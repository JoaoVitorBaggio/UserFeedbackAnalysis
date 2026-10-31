import pytest

from user_feedback_analysis.baseline.classifier import (
    preprocess_cardiff_twitter,
    uses_tweet_preprocess,
)
from user_feedback_analysis.baseline.mapping import (
    ALTERNATIVE_MODEL_ID,
    DEFAULT_MODEL_ID,
    pipeline_label_to_sentiment,
    sentiment_to_label,
    validate_id2label,
)
from user_feedback_analysis.data.load import as_bool, as_optional_int, corpus_from_mappings
from user_feedback_analysis.labels import Label, label_from_flags, normalize_label


@pytest.mark.parametrize(
    ("satisfaction", "dissatisfaction", "expected"),
    [
        (True, False, Label.SAT),
        (False, True, Label.DSAT),
        (False, False, Label.NO_SIGNAL),
        (True, True, None),
    ],
)
def test_flag_map(satisfaction, dissatisfaction, expected):
    assert label_from_flags(satisfaction, dissatisfaction) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SAT", Label.SAT),
        ("dsat", Label.DSAT),
        ("sem sinal", Label.NO_SIGNAL),
        ("sem_sinal", Label.NO_SIGNAL),
        ("no-signal", Label.NO_SIGNAL),
        ("NO_SIGNAL", Label.NO_SIGNAL),
        (Label.SAT, Label.SAT),
    ],
)
def test_label_aliases(raw, expected):
    assert normalize_label(raw) is expected


def test_unknown_label_fails():
    with pytest.raises(ValueError):
        normalize_label("positivo")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("FALSE", False),
        (1.0, True),
        (0.0, False),
        (None, None),
    ],
)
def test_bool_parser(value, expected):
    assert as_bool(value) is expected


def test_sentiment_projects_onto_three_classes():
    assert sentiment_to_label("Positive") is Label.SAT
    assert sentiment_to_label("negative") is Label.DSAT
    assert sentiment_to_label("NEUTRAL") is Label.NO_SIGNAL
    with pytest.raises(ValueError, match="duas classes"):
        sentiment_to_label("LABEL_0")


def test_checkpoint_must_expose_neutral():
    mapping = validate_id2label(
        {0: "Negative", 1: "Neutral", 2: "Positive"},
        DEFAULT_MODEL_ID,
    )
    assert mapping == {0: "negative", 1: "neutral", 2: "positive"}
    with pytest.raises(ValueError, match="duas classes"):
        validate_id2label({0: "negative", 1: "positive"}, "example/two-way")


def test_alternative_checkpoint_exposes_three_classes_without_tweet_preprocess():
    assert uses_tweet_preprocess(DEFAULT_MODEL_ID)
    assert not uses_tweet_preprocess(ALTERNATIVE_MODEL_ID)
    mapping = validate_id2label(
        {0: "negative", 1: "neutral", 2: "positive"},
        ALTERNATIVE_MODEL_ID,
    )
    assert set(mapping.values()) == {"negative", "neutral", "positive"}


def test_legacy_cardiff_label_ids_are_explicit():
    mapping = validate_id2label(
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        "cardiffnlp/twitter-roberta-base-sentiment",
    )
    assert mapping[0] == "negative"
    assert mapping[1] == "neutral"
    assert mapping[2] == "positive"
    with pytest.raises(ValueError):
        validate_id2label(
            {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
            "example/unknown-label-names",
        )
    assert (
        pipeline_label_to_sentiment("LABEL_0", {0: "negative", 1: "neutral", 2: "positive"})
        == "negative"
    )


def test_cardiff_preprocess_matches_the_model_card():
    assert preprocess_cardiff_twitter("@joao see https://example.com/a") == "@user see http"


def test_numpy_style_scalars_coerce():
    class _Bool:
        def item(self):
            return True

    class _Int:
        def item(self):
            return 7

    assert as_bool(_Bool()) is True
    assert as_optional_int(_Int()) == 7


def test_jsonl_roundtrip(tiny_path, tmp_path):
    from user_feedback_analysis.data.load import load_jsonl, write_jsonl
    from user_feedback_analysis.data.split import prepare_examples

    original = load_jsonl(tiny_path)
    target = tmp_path / "copy.jsonl"
    write_jsonl(original, target)
    copied = load_jsonl(target)
    assert [row.conversation_id for row in copied.rows] == [
        row.conversation_id for row in original.rows
    ]
    assert [row.dissatisfaction for row in copied.rows] == [
        row.dissatisfaction for row in original.rows
    ]
    left = prepare_examples(original, role="User")
    right = prepare_examples(copied, role="User")
    assert [(item.example_id, item.label) for item in left.examples] == [
        (item.example_id, item.label) for item in right.examples
    ]


def test_official_column_typos_and_aliases():
    corpus = corpus_from_mappings(
        [
            {
                "UtterranceId": 0,
                "Role": "User",
                "Content": "a",
                "Satisfaction": True,
                "Disatisfaction": False,
            },
            {
                "UtterranceId": 0,
                "Role": "User",
                "Content": "b",
                "Satisfaction": "false",
                "Dissatisfaction": "true",
            },
        ],
        source="mem",
    )
    assert [row.conversation_id for row in corpus.rows] == [0, 1]
    assert corpus.rows[0].dissatisfaction is False
    assert corpus.rows[1].dissatisfaction is True
    assert corpus.rows[1].satisfaction is False
