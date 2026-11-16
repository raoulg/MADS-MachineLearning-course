"""What every design script of the course shares: optional stages, and the check that a
search space searches them.

A design is a supergraph: every block the search may build, written once as a method of
an `nn.Module` (a stage). A block the search may leave out is a choice in `forward`
(`if self.config.skip:`), and `@option("skip")` on the stage names the axis of the search
space that decides it. The config a trial draws picks the subgraph that is built.

    from hpo.designs import option

    class Head(nn.Module):
        @option("skip")
        @jaxtyped(typechecker=beartype)
        def skip(self, x: Float[Tensor, "batch {self.dim}"]) -> ...: ...

`check_space` refuses a search space that leaves an option out or holds an axis nothing
reads; `combinations` and `trace` build every subgraph and show the shape after every
stage, which is what running a design script prints.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch import Tensor, nn


def option[F: Callable[..., Any]](axis: str) -> Callable[[F], F]:
    """Mark a stage as one the search may leave out, decided by the axis `axis` of the
    search space. The stage stays as it is; the axis name is kept on it as `__option__`."""

    def mark(stage: F) -> F:
        vars(stage)["__option__"] = axis
        return stage

    return mark


def axis_of(stage: object) -> str | None:
    """The axis `option` named on `stage`, or None for a stage that is always built."""
    try:
        return vars(stage).get("__option__")
    except TypeError:  # an object without attributes of its own
        return None


def options(cls: type) -> list[str]:
    """The axis names of the options of a design, in the order its stages are written."""
    return [axis for stage in vars(cls).values() if (axis := axis_of(stage))]


def stages(cls: type) -> list[str]:
    """The names of the stages of a design: its methods other than `forward`."""
    return [
        name
        for name, stage in vars(cls).items()
        if callable(stage) and not name.startswith("_") and name != "forward"
    ]


def drawn[C](cls: type[C], config: Mapping[str, Any]) -> C:
    """The choices of the dataclass `cls` in a config a trial drew; the rest of the
    config (the dataset, the learning rate, ...) is the trainer's."""
    if not is_dataclass(cls):
        raise TypeError(f"drawn: {cls.__name__} is not a dataclass")
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in config.items() if k in names})


def choices(cls: type) -> list[str]:
    """The field names of a design's config dataclass: the axes the design reads."""
    if not is_dataclass(cls):
        raise TypeError(f"choices: {cls.__name__} is not a dataclass")
    return [f.name for f in fields(cls)]


def check_space(
    space: Mapping[str, Any],
    options: Iterable[str],
    design: Iterable[str],
    trainer: Iterable[str],
) -> None:
    """A `ValueError` unless every option of the design is an axis of `space`, and every
    key of `space` is known to the design (`design`, the names of its choices) or to the
    trainer (`trainer`, the keys the lesson's harness reads)."""
    for axis in options:
        if axis not in space:
            raise ValueError(
                f"SPACE has no axis {axis!r}, which designs.py marks @option({axis!r}): "
                f"add it to SPACE, as {axis!r}: Choice([False, True]) to search it or "
                f"{axis!r}: False to leave the stage out of every trial"
            )
    used = set(design) | set(trainer)
    for key in space:
        if key not in used:
            raise ValueError(
                f"SPACE has the axis {key!r}, which neither the design nor the trainer "
                f"uses: give Config in designs.py a field {key!r} and a stage that reads "
                "it, or take it out of SPACE"
            )


def combinations(cls: type) -> Iterator[dict[str, bool]]:
    """Every combination of the options of a design, each option on or off."""
    axes = options(cls)
    for values in itertools.product([False, True], repeat=len(axes)):
        yield dict(zip(axes, values, strict=True))


def trace(model: nn.Module, example: Any) -> list[tuple[str, tuple[int, ...]]]:
    """The stages `model.forward(example)` runs, in order, each with the shape it gives.
    `example` is one batch input: a tensor, or a tuple of tensors for a model with more
    than one stream."""
    first = example[0] if isinstance(example, (tuple, list)) else example
    seen: list[tuple[str, tuple[int, ...]]] = [("input", tuple(first.shape))]

    def watch(name: str, stage: Callable[..., Tensor]) -> Callable[..., Tensor]:
        def watched(*args: Any, **kwargs: Any) -> Tensor:
            out = stage(*args, **kwargs)
            shape = out[0].shape if isinstance(out, (tuple, list)) else out.shape
            seen.append((name, tuple(shape)))
            return out

        return watched

    names = stages(type(model))
    for name in names:
        # an attribute on the instance comes before the method of the class
        object.__setattr__(model, name, watch(name, getattr(model, name)))
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(example)
    finally:
        model.train(training)
        for name in names:
            object.__delattr__(model, name)
    return seen


def show(
    cls: type,
    build: Callable[[dict[str, bool]], nn.Module],
    example: Any,
) -> None:
    """Print, for every combination of the options of the design `cls`, its number of
    params and the shape after every stage. `build(chosen)` makes the model a
    combination chooses."""
    for chosen in combinations(cls):
        model = build(chosen)
        n_params = sum(p.numel() for p in model.parameters())
        label = (
            "  ".join(f"{axis}={on}" for axis, on in chosen.items()) or "(no options)"
        )
        print(f"{label}   ({n_params:,} params)")
        for name, shape in trace(model, example):
            print(f"  {name:<12} {shape}")
        print()
