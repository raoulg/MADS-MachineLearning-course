"""What every design script of the course shares: optional stages, and the design that
reads them.

A design is a supergraph: every block the search may build, written once as a method of
an `nn.Module` (a stage). A block the search may leave out is a choice in `forward`
(`if self.config.skip:`), and `@option("skip")` on the stage names the axis of the search
space that decides it. The config a trial draws picks the subgraph that is built.

    from hpo.designs import Design, option

    class Head(nn.Module):
        @option("skip")
        @jaxtyped(typechecker=beartype)
        def skip(self, x: Float[Tensor, "batch {self.dim}"]) -> ...: ...

    DESIGN = Design(Head, Config)

`Design.combinations` and `Design.trace` build every subgraph and show the shape after
every stage, which is what running a design script prints.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields, is_dataclass

import torch
from pydantic import BaseModel
from torch import Tensor, nn
from vectormesh.types import TensorInput

from hpo.space import Param


def option[F: Callable[..., object]](axis: str) -> Callable[[F], F]:
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


class Step(BaseModel):
    """One stage a forward pass ran, and the shape it gave."""

    stage: str
    shape: tuple[int, ...]


class Design[M: nn.Module, C]:
    """The design class `module` and the config dataclass `config` that holds its
    choices."""

    def __init__(self, module: type[M], config: type[C]) -> None:
        if not is_dataclass(config):
            raise TypeError(f"Design: {config.__name__} is not a dataclass")
        self.module = module
        self.config = config

    def options(self) -> list[str]:
        """The axis names of the options, in the order the stages are written."""
        return [
            axis for stage in vars(self.module).values() if (axis := axis_of(stage))
        ]

    def stages(self) -> list[str]:
        """The names of the stages: the module's methods other than `forward`."""
        return [
            name
            for name, stage in vars(self.module).items()
            if callable(stage) and not name.startswith("_") and name != "forward"
        ]

    def choices(self) -> list[str]:
        """The field names of the config: the axes the design reads."""
        return [f.name for f in fields(self.config)]

    def drawn(self, config: Mapping[str, Param]) -> C:
        """The choices of the design in a config a trial drew; the rest of the config
        (the dataset, the learning rate, ...) is the trainer's."""
        names = set(self.choices())
        return self.config(**{k: v for k, v in config.items() if k in names})

    def combinations(self) -> Iterator[dict[str, bool]]:
        """Every combination of the options, each option on or off."""
        axes = self.options()
        for values in itertools.product([False, True], repeat=len(axes)):
            yield dict(zip(axes, values, strict=True))

    def trace(self, model: M, example: TensorInput) -> list[Step]:
        """The stages `model(example)` runs, in order, each with the shape it gives.
        `example` is one batch input: a tensor, or a tuple of tensors for a model with
        more than one stream. The model is left as it was found."""
        first = example[0] if isinstance(example, tuple | list) else example
        seen = [Step(stage="input", shape=tuple(first.shape))]

        def watch(name: str, stage: Callable[..., Tensor]) -> Callable[..., Tensor]:
            def watched(*args: object, **kwargs: object) -> Tensor:
                out = stage(*args, **kwargs)
                shape = out[0].shape if isinstance(out, tuple | list) else out.shape
                seen.append(Step(stage=name, shape=tuple(shape)))
                return out

            return watched

        names = self.stages()
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

    def show(self, build: Callable[[dict[str, bool]], M], example: TensorInput) -> None:
        """Print, for every combination of the options, its number of params and the
        shape after every stage. `build(chosen)` makes the model a combination chooses."""
        for chosen in self.combinations():
            model = build(chosen)
            n_params = sum(p.numel() for p in model.parameters())
            label = (
                "  ".join(f"{axis}={on}" for axis, on in chosen.items())
                or "(no options)"
            )
            print(f"{label}   ({n_params:,} params)")
            for step in self.trace(model, example):
                print(f"  {step.stage:<12} {step.shape}")
            print()
