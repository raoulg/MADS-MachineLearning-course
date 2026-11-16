"""hpo.metrics: every metric from one confusion matrix, and the spread over repeats."""

import pytest
import torch
from torch import nn

from hpo.metrics import Scorer, scores, summarise, value

CLASSES = ["N", "S", "V", "F", "Q"]


def test_every_metric_of_a_small_confusion():
    # N: 2 of 3 right, S: 1 of 1, V: 0 of 1 (predicted as N); F and Q have no test rows
    m = scores([0, 0, 0, 1, 2], [0, 0, 1, 1, 0], CLASSES)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["recall"] == {"N": pytest.approx(2 / 3, abs=1e-5), "S": 1.0, "V": 0.0}
    assert m["macro_recall"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert m["precision"]["N"] == pytest.approx(2 / 3, abs=1e-5)
    assert m["precision"]["S"] == pytest.approx(1 / 2)
    assert m["f1"]["S"] == pytest.approx(2 / 3, abs=1e-5)


def test_a_class_without_test_rows_has_no_recall_and_is_left_out_of_the_mean():
    m = scores([0, 0, 1], [0, 0, 4], CLASSES)
    assert set(m["recall"]) == {"N", "S"}
    assert set(m["f1"]) == {"N", "S"}
    assert m["macro_recall"] == pytest.approx(
        0.5
    )  # (1 + 0) / 2, not (1 + 0 + 0 + 0 + 0) / 5
    # a class that is predicted keeps its precision, even without test rows
    assert m["precision"]["Q"] == 0.0


def test_macro_recall_sees_the_rare_class_that_accuracy_hides():
    truth = [0] * 95 + [3] * 5
    always_n = [0] * 100
    m = scores(truth, always_n, CLASSES)
    assert m["accuracy"] == pytest.approx(0.95)
    assert m["macro_recall"] == pytest.approx(0.5)


def test_summarise_gives_the_mean_and_the_spread_in_the_same_shape():
    a = scores([0, 1], [0, 1], CLASSES)
    b = scores([0, 1], [0, 0], CLASSES)
    mean, std = summarise([a, b])
    assert mean["accuracy"] == pytest.approx(0.75)
    assert std["accuracy"] == pytest.approx(0.25)
    assert mean["recall"]["S"] == pytest.approx(0.5)
    assert set(std["recall"]) == set(a["recall"])


def test_value_reads_a_scalar_or_one_class():
    m = scores([0, 1], [0, 1], CLASSES)
    assert value(m, "accuracy") == 1.0
    assert value(m, "recall/S") == 1.0
    with pytest.raises(KeyError, match="recall/Q"):
        value(m, "recall/Q")


def test_the_scorer_reads_one_hot_targets_and_restores_the_mode():
    model = nn.Linear(2, 3).train()
    x = torch.randn(6, 2)
    y = nn.functional.one_hot(model(x).argmax(1), 3).float()
    m = Scorer([(x, y)], ["a", "b", "c"])(model)
    assert m["accuracy"] == 1.0
    assert model.training
