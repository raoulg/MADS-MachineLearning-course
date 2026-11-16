"""The design and the space check of lesson 1's exercise folder, without data."""

import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "notebooks/1_pytorch_intro/exercises")
)

from designs import Config, Head  # ty: ignore[import-not-found]  (the exercise folder is put on sys.path above)
from harness import HARNESS  # ty: ignore[import-not-found]

from hpo.designs import axis_of, check_space, combinations, option, options, trace

DESIGN = [f.name for f in fields(Config)]


@pytest.mark.parametrize("chosen", list(combinations(Head)), ids=str)
@pytest.mark.parametrize("norm", ["none", "batch", "layer"])
def test_every_combination_gives_batch_by_classes(chosen, norm):
    config = Config(hidden_dim=16, n_hidden_layers=2, norm=norm, dropout=0.1, **chosen)
    model = Head(config, dim=12, n_classes=5)
    assert model(torch.randn(4, 12)).shape == (4, 5)
    ran = [name for name, _ in trace(model, torch.randn(4, 12))]
    for axis, on in chosen.items():
        assert (axis in ran) == on


def test_options_names_the_axes_of_the_design():
    assert options(Head) == ["skip", "gate"]
    assert len(list(combinations(Head))) == 4


def test_option_leaves_the_stage_as_it_is():
    def stage(x):
        return x + 1

    assert option("width")(stage) is stage
    assert axis_of(stage) == "width"


SPACE = {
    "dataset": "dtd",
    "encoder": "dinov2-small",
    "hidden_dim": 64,
    "skip": False,
    "gate": True,
    "lr": 1e-3,
    "epochs": 1,
}


def test_check_space_takes_a_space_that_searches_every_option():
    check_space(SPACE, options(Head), DESIGN, HARNESS.trainer)


def test_check_space_refuses_a_missing_option():
    space = {k: v for k, v in SPACE.items() if k != "gate"}
    with pytest.raises(ValueError, match="no axis 'gate'"):
        check_space(space, options(Head), DESIGN, HARNESS.trainer)


def test_check_space_refuses_an_unused_axis():
    with pytest.raises(ValueError, match="'weight_decay', which neither"):
        check_space(
            {**SPACE, "weight_decay": 1e-4}, options(Head), DESIGN, HARNESS.trainer
        )


def test_drawn_keeps_the_choices_of_the_design_and_leaves_the_trainers():
    config = Config.drawn(
        {"hidden_dim": 48, "skip": True, "lr": 1e-3, "dataset": "dtd"}
    )
    assert (config.hidden_dim, config.skip, config.gate) == (48, True, False)


@pytest.mark.parametrize("axis", options(Head))
def test_an_option_changes_what_the_head_computes(axis):
    torch.manual_seed(0)
    model = Head(Config(hidden_dim=16, **{axis: True}), dim=12, n_classes=5).eval()
    embed = torch.randn(4, 12)
    with_it = model(embed)
    object.__setattr__(model, "config", Config(hidden_dim=16))
    assert not torch.allclose(with_it, model(embed))


def test_trace_leaves_the_model_as_it_found_it():
    model = Head(Config(hidden_dim=16, skip=True), dim=12, n_classes=5).train()
    trace(model, torch.randn(4, 12))
    assert model.training and "skip" not in vars(model)


def test_a_head_without_a_hidden_block_is_refused_by_name():
    with pytest.raises(ValueError, match="n_hidden_layers must be at least 1"):
        Head(Config(n_hidden_layers=0), dim=12, n_classes=5)
