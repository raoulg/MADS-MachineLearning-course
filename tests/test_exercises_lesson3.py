"""The design, the control width, the cut of the data and the stopping rule of lesson 3's
exercise folder, without the hub."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from hpo.measure import Architecture, Confusion, Fit

FOLDER = (
    Path(__file__).resolve().parents[1] / "notebooks/3_scaling_and_search/exercises"
)


def load(name: str):
    # under a name of its own: other lessons' folders have modules called designs and
    # harness too
    spec = importlib.util.spec_from_file_location(
        f"lesson3_{name}", FOLDER / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


designs = load("designs")
harness = load("harness")
DESIGN, HEAD, Config, Mix = designs.DESIGN, designs.HEAD, designs.Config, designs.Mix
SMALL = {"dim": 384, "height": 16, "breadth": 24, "n_classes": 321}


def images(n: int = 4, grid: bool = False) -> torch.Tensor:
    return torch.randn(n, 2, 16, 24) if grid else torch.randn(n, 2, 384)


@pytest.mark.parametrize("head", list(HEAD.combinations()), ids=str)
@pytest.mark.parametrize("chosen", list(DESIGN.combinations()), ids=str)
def test_every_combination_gives_batch_by_classes(chosen, head):
    model = Mix(Config(width=8, n_experts=3, **chosen, **head), **SMALL)
    assert model(images()).shape == (4, 321)
    ran = [step.stage for step in DESIGN.trace(model, images())]
    for axis, on in chosen.items():
        assert (axis in ran) == on


@pytest.mark.parametrize("kernel", [False, True])
@pytest.mark.parametrize("input_", ["cls", "mean", "both"])
def test_both_forms_of_the_loader_give_the_same_scores(input_, kernel):
    model = Mix(Config(width=8, input=input_, kernel=kernel), **SMALL).eval()
    flat = images()
    assert torch.allclose(model(flat), model(harness.fold(flat)))


def test_the_base_encoder_folds_to_its_own_grid():
    model = Mix(
        Config(width=8, kernel=True), dim=768, height=24, breadth=32, n_classes=5
    )
    assert model(torch.randn(3, 2, 24, 32)).shape == (3, 5)
    assert harness.fold(torch.zeros(3, 2, 768)).shape == (3, 2, 24, 32)


def test_options_name_the_axes_of_the_design():
    assert DESIGN.options() == ["kernel", "moe"]
    assert HEAD.options() == ["skip", "gate"]


@pytest.mark.parametrize("axis", ["input", "skip", "gate"])
def test_a_choice_changes_what_the_model_computes(axis):
    torch.manual_seed(0)
    value = "mean" if axis == "input" else True
    model = Mix(Config(width=8, **{axis: value}), **SMALL).eval()
    x = images()
    with_it = model(x)
    object.__setattr__(model, "config", Config(width=8))
    object.__setattr__(model, "rows", designs.ROWS["cls"])
    object.__setattr__(model.dense, "config", Config(width=8))
    assert not torch.allclose(with_it, model(x))


def test_a_head_without_a_hidden_layer_is_refused_by_name():
    with pytest.raises(ValueError, match="depth must be at least 1"):
        Mix(Config(depth=0), **SMALL)


@pytest.mark.parametrize(("d_in", "control"), [(64, 9), (384, 12)])
def test_the_control_width_is_the_one_of_3_1(d_in, control):
    assert designs.control_width(8, 8, d_in, 321) == control


def test_the_moe_and_its_control_cost_about_the_same_flops():
    moe = Mix(Config(width=8, moe=True), **SMALL)
    control = Mix(Config(width=designs.control_width(8, 8, 384, 321)), **SMALL)
    mine, its = (
        Architecture.of(m, torch.zeros(2, 2, 384)).flops_per_image
        for m in (moe, control)
    )
    assert mine == 2 * (384 * 8 + 384 * 8 + 8 * 321)  # router and one head
    assert its is not None
    assert abs(mine - its) / mine < 0.05


@pytest.fixture
def mix(monkeypatch):
    """A mix of 3 domains, 6 classes of 40 rows, as `harness.load` gives it."""
    label = np.repeat(np.arange(6), 40)
    rows = harness.Rows(
        embed=np.random.default_rng(0).normal(size=(240, 2, 384)).astype(np.float32),
        label=label,
        domain=label // 2,
    )
    monkeypatch.setattr(harness, "load", lambda encoder, split="train": rows)
    monkeypatch.setattr(harness, "DOMAINS", ["a", "b", "c"])
    harness.order.cache_clear()
    yield rows
    harness.order.cache_clear()


def test_a_share_is_stratified_nested_and_never_in_validation(mix):
    small, valid = harness.cut("fake", 0.25)
    large, valid_too = harness.cut("fake", 0.5)
    assert np.array_equal(valid, valid_too)
    assert np.bincount(mix.label[valid]).tolist() == [4] * 6  # 10% of every class
    assert np.bincount(mix.label[small]).tolist() == [9] * 6  # 25% of the other 36
    assert set(small) <= set(large)
    assert not set(large) & set(valid)


def test_a_dropped_domain_is_left_out_of_train_and_validation(mix):
    train, valid = harness.cut("fake", 1.0, drop_domain="b")
    assert 1 not in set(mix.domain[train]) | set(mix.domain[valid])
    with pytest.raises(KeyError, match="no domain"):
        harness.cut("fake", 1.0, drop_domain="zz")


def test_the_loader_standardises_on_the_train_rows_and_folds_the_grid(mix):
    train, valid = harness.tensors("fake", 1.0, form="grid")
    assert train.tensors[0].shape[1:] == (2, 16, 24)
    flat = train.tensors[0].flatten(2)
    assert torch.allclose(flat.mean(0), torch.zeros(2, 384), atol=1e-5)
    assert len(valid) == 24


def fit_at(accuracy: float) -> Fit:
    right = round(100 * accuracy)
    truth = torch.zeros(100, dtype=torch.long)
    guess = torch.cat([torch.zeros(right), torch.ones(100 - right)]).long()
    return Fit(confusion=Confusion.of(truth, guess, ["a", "b"]))


def test_patience_stops_after_epochs_without_a_better_score_and_keeps_the_best():
    model = torch.nn.Linear(2, 2)
    patience = harness.Patience(model, patience=2)
    stops = [patience.see(e, fit_at(a)) for e, a in enumerate([0.5, 0.7, 0.6, 0.65])]
    assert stops == [False, False, False, True]
    assert patience.epoch == 2
    assert patience.kept().scores().accuracy == pytest.approx(0.7)
