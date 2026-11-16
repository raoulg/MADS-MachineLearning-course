"""The design, the loss weights and the space check of lesson 2's exercise folder, without
data."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from hpo.search import Space

FOLDER = (
    Path(__file__).resolve().parents[1]
    / "notebooks/2_kernels_state_and_gates/exercises"
)


def load(name: str):
    # under a name of its own: lesson 1's folder has modules called designs and harness too
    spec = importlib.util.spec_from_file_location(
        f"lesson2_{name}", FOLDER / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


designs = load("designs")
harness = load("harness")
DESIGN, Config, Beats = designs.DESIGN, designs.Config, designs.Beats


def beats(n: int = 4) -> torch.Tensor:
    return torch.randn(n, designs.STEPS, 1)


@pytest.mark.parametrize("chosen", list(DESIGN.combinations()), ids=str)
def test_every_combination_gives_batch_by_classes(chosen):
    config = Config(width=8, depth=2, **chosen)
    model = Beats(config, n_classes=3)
    assert model(beats()).shape == (4, 3)
    ran = [step.stage for step in DESIGN.trace(model, beats())]
    for axis, on in chosen.items():
        assert (axis in ran) == on


@pytest.mark.parametrize("downsample", ["pool", "stride"])
@pytest.mark.parametrize("norm", ["none", "batch", "layer"])
@pytest.mark.parametrize("depth", [1, 5])
def test_depth_downsample_and_norm_give_batch_by_classes(depth, norm, downsample):
    config = Config(width=8, depth=depth, norm=norm, downsample=downsample, skip=True)
    assert Beats(config, n_classes=3)(beats()).shape == (4, 3)


def test_options_names_the_axes_of_the_design():
    assert DESIGN.options() == ["parallel", "fold", "skip", "excite", "state", "gate"]


@pytest.mark.parametrize("axis", DESIGN.options())
def test_an_option_changes_what_the_model_computes(axis):
    torch.manual_seed(0)
    model = Beats(Config(width=8, depth=2, **{axis: True}), n_classes=3).eval()
    x = beats()
    with_it = model(x)
    object.__setattr__(model, "config", Config(width=8, depth=2))
    assert not torch.allclose(with_it, model(x))


def test_a_model_without_a_block_is_refused_by_name():
    with pytest.raises(ValueError, match="depth must be at least 1"):
        Beats(Config(depth=0), n_classes=3)


def test_weights_run_from_plain_to_balanced():
    counts = np.array([900, 30, 70])
    assert torch.allclose(harness.weights(counts, 0.0), torch.ones(3))
    balanced = harness.weights(counts, 1.0)
    # every class weighs as much in the loss: weight times count is the same for each
    assert np.allclose(balanced.numpy() * counts, balanced[0].item() * counts[0])


def check(space: dict) -> None:
    Space(space).check(DESIGN.options(), DESIGN.choices(), harness.HARNESS.trainer)


def test_the_handed_out_space_searches_every_option():
    source = (FOLDER / "study.py").read_text()
    for axis in DESIGN.options():
        assert f'"{axis}":' in source


def test_check_space_refuses_an_unused_axis():
    space = {axis: False for axis in DESIGN.options()} | {"dataset": "mitbih"}
    check(space)
    with pytest.raises(ValueError, match="'weight_decay', which neither"):
        check({**space, "weight_decay": 1e-4})
