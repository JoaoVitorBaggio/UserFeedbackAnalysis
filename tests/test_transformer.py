import pytest

from user_feedback_analysis.labels import LABELS, Label
from user_feedback_analysis.records import Example
from user_feedback_analysis.transformer.dataset import LABEL_TO_ID, SatisfactionDataset
from user_feedback_analysis.transformer.loss import LOSS_NAME, N_CLASSES, build_loss
from user_feedback_analysis.transformer.model import SatisfactionModel
from user_feedback_analysis.transformer.train import train_transformer


def test_label_ids_are_the_three_classes():
    assert N_CLASSES == 3
    assert LOSS_NAME == "categorical_cross_entropy"
    assert set(LABEL_TO_ID) == set(LABELS)
    assert set(LABEL_TO_ID.values()) == {0, 1, 2}
    assert LABEL_TO_ID[Label.SAT] == 0
    assert LABEL_TO_ID[Label.DSAT] == 1
    assert LABEL_TO_ID[Label.NO_SIGNAL] == 2


def test_dataset_yields_text_and_index():
    example = Example(
        example_id="r0",
        conversation_id=0,
        text="Thanks, this is perfect.",
        label=Label.SAT,
        role="User",
        source_index=0,
    )
    dataset = SatisfactionDataset([example])
    assert len(dataset) == 1
    item = dataset[0]
    assert item["text"] == example.text
    assert item["label"] == 0
    assert item["label_name"] == "SAT"


def test_loss_helper_rejects_class_weights_before_training():
    with pytest.raises(ValueError, match="não pondera"):
        build_loss(class_weights=[1.0, 1.0, 1.0])


def test_training_loop_and_forward_stay_unimplemented():
    with pytest.raises(NotImplementedError, match="Nikolas"):
        train_transformer([])
    model = SatisfactionModel("a definir")
    assert model.num_labels == 3
    with pytest.raises(NotImplementedError, match="Nikolas"):
        model.forward({"text": "x"})


def test_build_loss_is_unweighted_cross_entropy():
    torch = pytest.importorskip("torch")
    criterion = build_loss()
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.weight is None
