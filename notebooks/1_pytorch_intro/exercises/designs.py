"""The design of your study: the head as a supergraph, one typed stage per operation.

A design is every block the search may build, written once. Each block the search may
leave out is a choice in `forward` (`if self.config.skip:`), and the config a trial draws
picks the subgraph that is built. `@option("skip")` on a stage names the axis of the
search space that decides it; `study.py` refuses a space that does not search every
option of the design.

Read it in atlas before you train anything: the shape on every arrow, and where a choice
leaves a stage out.

    atlas-code-nightly view notebooks/1_pytorch_intro/exercises

Run it from the root of the course repository to build every combination of the options
and see the shape after every stage:

    uv run notebooks/1_pytorch_intro/exercises/designs.py

To design your own:

1. Add a stage: a method of `Head` whose signature says the shape it takes and the shape
   it gives, in the widths the model was built with, under `@jaxtyped(typechecker=beartype)`.
   Build its layers in `__init__`, only when the config chooses it.
2. If the search may leave it out, mark it `@option("<axis>")`, give `Config` a field
   `<axis>: bool = False`, and call it in `forward` under `if self.config.<axis>:`.
3. Add the axis to `SPACE` in `study.py`: `"<axis>": Choice([False, True])`.
"""

import itertools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from vectormesh.components import Gate, GaussianNoise, Norm, Projection, Serial

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


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


@dataclass(frozen=True)
class Config:
    """The choices of the design. The search space draws them (`SPACE` in `study.py`);
    a field the space leaves out keeps its default here."""

    n_hidden_layers: int = 1
    hidden_dim: int = 256
    activation: str = "gelu"  # a key of ACTIVATIONS
    norm: Literal["none", "batch", "layer"] = "none"
    dropout: float = 0.0
    sigma: float = 0.0  # Gaussian noise on the input vectors, relative to their spread
    skip: bool = False  # a skip connection around the hidden block
    gate: bool = False  # a gate on the hidden block's output

    @classmethod
    def drawn(cls, config: Mapping[str, Any]) -> "Config":
        """The choices of this design in a config a trial drew; the rest of the config
        (the dataset, the learning rate, ...) is the trainer's."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in names})


class Head(nn.Module):
    """The head handed out in lesson 1, as a supergraph: input noise, a hidden block, a
    skip around it and a gate on it, then the classifier. Each stage is a method whose
    signature states the shape it takes and the shape it gives, in terms of the widths
    this model was built with, and `@jaxtyped` checks that statement on every call."""

    skip_projection: Projection
    gating: Gate

    def __init__(self, config: Config, dim: int, n_classes: int) -> None:
        super().__init__()
        self.config = config
        self.dim = dim
        self.hidden_dim = config.hidden_dim
        self.n_classes = n_classes
        self.noise = GaussianNoise(sigma=config.sigma)
        layers: list[nn.Module] = []
        width = dim
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(width, config.hidden_dim))
            if config.norm != "none":
                layers.append(Norm(config.hidden_dim, kind=config.norm))
            layers.append(ACTIVATIONS[config.activation]())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            width = config.hidden_dim
        self.block = Serial(layers)
        # an optional stage's layers exist only in the subgraph that uses them, so
        # n_params counts what the trial trained
        if config.skip:
            self.skip_projection = Projection(dim, config.hidden_dim)
        if config.gate:
            self.gating = Gate(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, n_classes)

    @jaxtyped(typechecker=beartype)
    def perturb(
        self, embed: Float[Tensor, "batch {self.dim}"]
    ) -> Float[Tensor, "batch {self.dim}"]:
        return self.noise(embed)

    @jaxtyped(typechecker=beartype)
    def hidden(
        self, noisy: Float[Tensor, "batch {self.dim}"]
    ) -> Float[Tensor, "batch {self.hidden_dim}"]:
        return self.block(noisy)

    @option("skip")
    @jaxtyped(typechecker=beartype)
    def skip(
        self, noisy: Float[Tensor, "batch {self.dim}"]
    ) -> Float[Tensor, "batch {self.hidden_dim}"]:
        return self.skip_projection(noisy)

    @option("gate")
    @jaxtyped(typechecker=beartype)
    def gate(
        self, hidden: Float[Tensor, "batch {self.hidden_dim}"]
    ) -> Float[Tensor, "batch {self.hidden_dim}"]:
        return self.gating(hidden)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, hidden: Float[Tensor, "batch {self.hidden_dim}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.classifier(hidden)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, embed: Float[Tensor, "batch {self.dim}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        noisy = self.perturb(embed)
        hidden = self.hidden(noisy)
        if self.config.skip:
            hidden = hidden + self.skip(noisy)
        if self.config.gate:
            hidden = self.gate(hidden)
        return self.classify(hidden)


def combinations(cls: type) -> Iterator[dict[str, bool]]:
    """Every combination of the options of a design, each option on or off."""
    axes = options(cls)
    for values in itertools.product([False, True], repeat=len(axes)):
        yield dict(zip(axes, values, strict=True))


def trace(model: nn.Module, embed: Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """The stages `model.forward(embed)` runs, in order, each with the shape it gives."""
    seen: list[tuple[str, tuple[int, ...]]] = [("embed", tuple(embed.shape))]

    def watch(name: str, stage: Callable[..., Tensor]) -> Callable[..., Tensor]:
        def watched(*args: Any) -> Tensor:
            out = stage(*args)
            seen.append((name, tuple(out.shape)))
            return out

        return watched

    for name in stages(type(model)):
        # an attribute on the instance comes before the method of the class
        object.__setattr__(model, name, watch(name, getattr(model, name)))
    try:
        model.eval()
        with torch.no_grad():
            model(embed)
    finally:
        for name in stages(type(model)):
            object.__delattr__(model, name)
    return seen


def main() -> None:
    dim, n_classes = 384, 47  # dinov2-small on DTD
    embed = torch.randn(2, dim)  # two rows are enough to trace every shape
    for chosen in combinations(Head):
        model = Head(Config.drawn(chosen), dim, n_classes)
        n_params = sum(p.numel() for p in model.parameters())
        label = "  ".join(f"{axis}={on}" for axis, on in chosen.items())
        print(f"{label}   ({n_params:,} params)")
        for name, shape in trace(model, embed):
            print(f"  {name:<10} {shape}")
        print()


if __name__ == "__main__":
    main()
