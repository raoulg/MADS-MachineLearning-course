"""The design of your study: the head as a supergraph, one typed stage per operation.

A design is every block the search may build, written once. Each block the search may
leave out is a choice in `forward` (`if self.config.skip:`), and the config a trial draws
picks the subgraph that is built. `@option("skip")` on a stage names the axis of the
search space that decides it; `study.py` refuses a space that does not search every
option of the design. `option` and `Design`, which reads them, are in `vectormesh.design`,
the same for every lesson.

Read it in atlas before you train anything: the shape on every arrow, and where a choice
leaves a stage out.

    atlas-code-nightly view notebooks/1_representation_and_head/exercises

Run it from the root of the course repository to build every combination of the options
and see the shape after every stage:

    uv run notebooks/1_representation_and_head/exercises/designs.py

To design your own:

1. Add a stage: a method of `Head` whose signature says the shape it takes and the shape
   it gives, in the widths the model was built with, under `@jaxtyped(typechecker=beartype)`.
   Build its layers in `__init__`, only when the config chooses it.
2. If the search may leave it out, mark it `@option("<axis>")`, give `Config` a field
   `<axis>: bool = False`, and call it in `forward` under `if self.config.<axis>:`.
3. Add the axis to `SPACE` in `study.py`: `"<axis>": Choice([False, True])`.
"""

from dataclasses import dataclass
from itertools import pairwise

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from vectormesh.components import (
    ActivationKind,
    Dense,
    Gate,
    GaussianNoise,
    NormKind,
    Projection,
    Serial,
)
from vectormesh.design import Design, option


@dataclass(frozen=True)
class Config:
    """The choices of the design. The search space draws them (`SPACE` in `study.py`);
    a field the space leaves out keeps its default here."""

    n_hidden_layers: int = 1
    hidden_dim: int = 256
    activation: ActivationKind = "gelu"
    norm: NormKind = "none"
    dropout: float = 0.0
    sigma: float = 0.0  # Gaussian noise on the input vectors, relative to their spread
    skip: bool = False  # a skip connection around the hidden block
    gate: bool = False  # a gate on the hidden block's output


class Head(nn.Module):
    """The head handed out in lesson 1, as a supergraph: input noise, a hidden block, a
    skip around it and a gate on it, then the classifier. Each stage is a method whose
    signature states the shape it takes and the shape it gives, in terms of the widths
    this model was built with, and `@jaxtyped` checks that statement on every call."""

    skip_projection: Projection
    gating: Gate

    def __init__(self, config: Config, dim: int, n_classes: int) -> None:
        super().__init__()
        if config.n_hidden_layers < 1:
            raise ValueError(
                "Head: n_hidden_layers must be at least 1; without a hidden block the "
                "head is the linear baseline, which study.py measures by itself"
            )
        self.config = config
        self.dim = dim
        self.hidden_dim = config.hidden_dim
        self.n_classes = n_classes
        self.noise = GaussianNoise(sigma=config.sigma)
        # a Dense is Linear, Norm, activation, Dropout; `norm="none"` and `dropout=0.0`
        # leave their part in as the identity
        widths = [dim] + [config.hidden_dim] * config.n_hidden_layers
        self.block = Serial(
            [
                Dense(
                    width_in,
                    width_out,
                    norm=config.norm,
                    activation=config.activation,
                    dropout=config.dropout,
                )
                for width_in, width_out in pairwise(widths)
            ]
        )
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


# The head and the config that switches its options, as the study reads them.
DESIGN = Design(Head, Config)


if __name__ == "__main__":
    # dinov2-small is 384 wide and DTD has 47 textures; two rows trace every shape
    DESIGN.show(torch.randn(2, 384), dim=384, n_classes=47)
