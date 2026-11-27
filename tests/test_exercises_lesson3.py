"""The design, the control width, the cut of the data and the stopping rule of lesson 3's
exercise folder, without the hub."""

import importlib.util
import json
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
    assert model(images(grid=True)).shape == (4, 321)
    ran = [step.stage for step in DESIGN.trace(model, images(grid=True))]
    for axis, on in chosen.items():
        assert (axis in ran) == on


@pytest.mark.parametrize("input_", ["cls", "mean", "both"])
def test_the_plain_head_reads_both_forms_of_the_loader_alike(input_):
    model = Mix(Config(width=8, input=input_), **SMALL).eval()
    flat = images()
    assert torch.allclose(model(flat), model(harness.fold(flat)))


def test_the_kernel_reads_the_grid_and_refuses_the_flat_form_by_name():
    model = Mix(Config(width=8, input="both", kernel=True), **SMALL)
    assert model(images(grid=True)).shape == (4, 321)
    with pytest.raises(ValueError, match="form='grid'"):
        model(images())


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


@pytest.mark.parametrize("width", [16, 64, 256])
def test_the_control_width_of_a_plain_head_is_its_own_width(width):
    model = Mix(Config(width=width), **SMALL)
    flops = Architecture.of(model, torch.zeros(2, 2, 384)).flops_per_image
    assert flops is not None
    assert designs.control_width(flops, 384, 321) == width


def test_the_moe_and_its_control_cost_about_the_same_flops():
    moe = Mix(Config(width=8, moe=True), **SMALL)
    mine = Architecture.of(moe, torch.zeros(2, 2, 384)).flops_per_image
    assert mine == 2 * (384 * 8 + 384 * 8 + 8 * 321)  # router and one head
    width = designs.control_width(mine, 384, 321)
    assert width == 12
    control = Mix(Config(width=width), **SMALL)
    its = Architecture.of(control, torch.zeros(2, 2, 384)).flops_per_image
    assert its is not None
    assert abs(mine - its) / mine < 0.05


def test_the_front_keeps_each_run_no_other_run_is_better_than():
    # d ties c on FLOPs and e ties a on accuracy: both are off; f equals a on both: on
    runs = [("a", 1, 0.5), ("b", 2, 0.4), ("c", 3, 0.7), ("d", 3, 0.6), ("e", 2, 0.5)]
    rows = [
        {"name": n, "flops": f, "accuracy": a} for n, f, a in [*runs, ("f", 1, 0.5)]
    ]
    assert [r["name"] for r in harness.front(rows)] == ["a", "f", "c"]


def test_the_front_reads_the_score_a_study_maximises():
    rows = [
        {"name": n, "flops": f, "objective": "macro_recall", "macro_recall": m}
        for n, f, m in [("a", 1, 0.5), ("b", 2, 0.4), ("c", 3, 0.7)]
    ]
    assert [r["name"] for r in harness.front(rows)] == ["a", "c"]


def fake_mix(monkeypatch, sizes: list[int]):
    """A mix of 3 domains and 6 classes of `sizes` rows, as `harness.load` gives it."""
    label = np.repeat(np.arange(6), sizes)
    rows = harness.Rows(
        embed=np.random.default_rng(0)
        .normal(size=(len(label), 2, 384))
        .astype(np.float32),
        label=label,
        domain=label // 2,
    )
    monkeypatch.setattr(harness, "load", lambda encoder, split="train": rows)
    monkeypatch.setattr(harness, "DOMAINS", ["a", "b", "c"])
    harness.order.cache_clear()
    return rows


@pytest.fixture
def mix(monkeypatch):
    """Six classes of 40 rows."""
    yield fake_mix(monkeypatch, [40] * 6)
    harness.order.cache_clear()


@pytest.fixture
def uneven(monkeypatch):
    """Classes of unequal size: 36, 36, 18, 18, 9 and 9 train rows after validation."""
    yield fake_mix(monkeypatch, [40, 40, 20, 20, 10, 10])
    harness.order.cache_clear()


def test_a_cap_is_nested_and_never_in_validation(mix):
    small, valid = harness.cut("fake", 9)
    large, valid_too = harness.cut("fake", 18)
    assert np.array_equal(valid, valid_too)
    assert np.bincount(mix.label[valid]).tolist() == [4] * 6  # 10% of every class
    assert np.bincount(mix.label[small]).tolist() == [9] * 6
    assert set(small) <= set(large)
    assert not set(large) & set(valid)


def test_a_cap_cuts_the_large_classes_and_keeps_the_small_ones_whole(uneven):
    capped, _ = harness.cut("fake", 12)
    assert np.bincount(uneven.label[capped]).tolist() == [12, 12, 12, 12, 9, 9]
    every, _ = harness.cut("fake")
    assert np.bincount(uneven.label[every]).tolist() == [36, 36, 18, 18, 9, 9]
    assert set(capped) <= set(every)


@pytest.mark.parametrize("cap", [0, -3, 0.25, 1.0, True, "80", 10**6])
def test_a_cap_that_is_no_count_of_rows_is_refused_by_name(mix, cap):
    with pytest.raises(ValueError, match="per_class is the most train rows"):
        harness.cut("fake", cap)


def test_a_config_with_a_share_is_refused_by_name(mix):
    with pytest.raises(KeyError, match="no axis 'share'"):
        harness.cap({"share": 0.3})


def test_every_run_logs_its_cap_and_its_train_rows(uneven, monkeypatch):
    monkeypatch.setattr(harness, "classes", lambda: list("abcdef"))
    whole = harness.HARNESS.data({})
    capped = harness.HARNESS.data({"per_class": 12})
    assert whole.dims["per_class"] == harness.PER_CLASS_ALL
    assert (whole.dims["n_train"], capped.dims["n_train"]) == (126, 66)
    assert whole.dims["steps_per_epoch"] == capped.dims["steps_per_epoch"]
    params = harness.HARNESS.params({"lr": 1e-3}, capped)
    assert (params["per_class"], params["n_train"]) == (12, 66)


def test_a_stored_run_without_its_cap_is_refused(tmp_path, monkeypatch):
    rows = [{"name": "w16 s0.1", "experiment": "E2", "share": 0.1}]
    lesson = tmp_path / "lesson"
    (lesson / "exercises").mkdir(parents=True)
    (lesson / "part1_stored.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(harness, "__file__", str(lesson / "exercises" / "harness.py"))
    with pytest.raises(ValueError, match="without 'per_class'"):
        harness.stored_rows("E2")


def test_a_dropped_domain_is_left_out_of_train_and_validation(mix):
    train, valid = harness.cut("fake", drop_domain="b")
    assert 1 not in set(mix.domain[train]) | set(mix.domain[valid])
    with pytest.raises(KeyError, match="no domain"):
        harness.cut("fake", drop_domain="zz")


def test_the_loader_standardises_on_the_train_rows_and_folds_the_grid(mix):
    train, valid = harness.tensors("fake", form="grid")
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
