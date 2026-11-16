"""hpo.measure.metrics: every metric from one confusion matrix, and the spread over repeats."""

import pytest
import torch
from jaxtyping import TypeCheckError
from torch import nn

from hpo.measure import Confusion, Repeats, Scalars, Scorer

CLASSES = ["N", "S", "V", "F", "Q"]


def scores_of(truth: list[int], guess: list[int]):
    return Confusion.of(torch.tensor(truth), torch.tensor(guess), CLASSES).scores()


def test_every_metric_of_a_small_confusion():
    # N: 2 of 3 right, S: 1 of 1, V: 0 of 1 (predicted as N); F and Q have no test rows
    m = scores_of([0, 0, 0, 1, 2], [0, 0, 1, 1, 0])
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.recall == {"N": pytest.approx(2 / 3), "S": 1.0, "V": 0.0}
    assert m.macro_recall == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert m.precision["N"] == pytest.approx(2 / 3)
    assert m.precision["S"] == pytest.approx(1 / 2)
    assert m.f1["S"] == pytest.approx(2 / 3)


def test_a_class_without_test_rows_has_no_recall_and_is_left_out_of_the_mean():
    m = scores_of([0, 0, 1], [0, 0, 4])
    assert set(m.recall) == {"N", "S"}
    assert set(m.f1) == {"N", "S"}
    assert m.macro_recall == pytest.approx(
        0.5
    )  # (1 + 0) / 2, not (1 + 0 + 0 + 0 + 0) / 5
    # a class that is predicted keeps its precision, even without test rows
    assert m.precision["Q"] == 0.0


def test_macro_recall_sees_the_rare_class_that_accuracy_hides():
    m = scores_of([0] * 95 + [3] * 5, [0] * 100)
    assert m.accuracy == pytest.approx(0.95)
    assert m.macro_recall == pytest.approx(0.5)


def test_labels_that_are_not_one_index_per_row_are_refused_by_their_shape():
    with pytest.raises(TypeCheckError):
        Confusion.of(torch.zeros(3, 5), torch.zeros(3, dtype=torch.long), CLASSES)


def test_a_confusion_is_square():
    with pytest.raises(ValueError, match="2 x 2"):
        Confusion(classes=["a", "b"], counts=[[1, 2]])


def test_repeats_give_the_mean_and_the_spread_in_the_shape_of_the_scores():
    a = Confusion.of(torch.tensor([0, 1]), torch.tensor([0, 1]), CLASSES)
    b = Confusion.of(torch.tensor([0, 1]), torch.tensor([0, 0]), CLASSES)
    runs = Repeats(runs=[a, b], seconds=2.0)
    assert runs.mean().accuracy == pytest.approx(0.75)
    assert runs.std().accuracy == pytest.approx(0.25)
    assert runs.mean().recall["S"] == pytest.approx(0.5)
    assert set(runs.std().recall) == set(a.scores().recall)
    assert list(runs.values("recall/S")) == [1.0, 0.0]
    assert runs.confusion().matrix()[0, 0] == 2 and runs.confusion().matrix()[1, 0] == 1


def test_a_metric_is_read_by_name():
    m = scores_of([0, 1], [0, 1])
    assert m.get("accuracy") == 1.0
    assert m.get("recall/S") == 1.0
    with pytest.raises(KeyError, match="recall/Q"):
        m.get("recall/Q")


def test_scalars_hold_one_value_per_repeat():
    runs = Scalars(accuracy=[0.5, 0.7])
    assert runs.logged() == ["accuracy"] and len(runs) == 2
    with pytest.raises(KeyError, match="macro_recall"):
        runs.values("macro_recall")
    with pytest.raises(ValueError, match="one value per repeat"):
        Scalars(accuracy=[0.5], macro_recall=[0.4, 0.3])


def test_the_scorer_reads_one_hot_targets_and_restores_the_mode():
    model = nn.Linear(2, 3).train()
    x = torch.randn(6, 2)
    y = nn.functional.one_hot(model(x).argmax(1), 3).float()
    m = Scorer([(x, y)], ["a", "b", "c"])(model).scores()
    assert m.accuracy == 1.0
    assert model.training
