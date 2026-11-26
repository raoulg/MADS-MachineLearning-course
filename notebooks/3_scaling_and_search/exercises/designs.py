"""The design of your study: a head over the two DINOv2 vectors of an image, as a
supergraph, one typed stage per operation.

An image of the mix is two vectors, `(batch, 2, dim)`: its CLS token and the mean of its
patch tokens (`dim` is 384 for `dinov2-small`, 768 for `dinov2-base`). With the trainer
axis `form="grid"` the loader gives every vector folded into a grid instead,
`(batch, 2, 16, 24)` for small; the kernel reads that grid, and the plain head flattens
either form. The design picks the vectors it reads, turns them into features
(flattened, or read by a kernel on the grid), and gives a score per class from a dense
head, or from a mixture of such heads behind a router.

Every block the search may leave out is a choice in `forward` (`if self.config.moe:`),
and `@option("moe")` on its stage names the axis of the search space that decides it.
The config a trial draws picks the subgraph that is built. `option` and `Design` are in
`vectormesh.design`, the same for every lesson.

The choices of this design, each a bet about the data:

- `input`: the CLS vector, the patch mean, or both side by side;
- `kernel`: the grid of the chosen vectors (`form="grid"`) read by a 2D kernel and a
  pool; folded 16 x 24, numbers 24 apart in the vector are neighbours, and the kernel
  bets that neighbours belong together, as lesson 2 bet on a folded beat;
- `width`, `depth`: the hidden layers of the head;
- `skip`: a projected skip around the hidden layers, `hidden + Projection(features)`
  (lesson 1's variant: the input and the hidden layer differ in width);
- `gate`: a gate on the output of the hidden layers (lesson 2);
- `norm`, `dropout`: a norm and dropout in every hidden layer; dropout sets a share of
  the hidden numbers to zero at random while training;
- `moe`: `n_experts` heads behind a router that runs one of them per image (3.1).

In a shape, `*form` stands for any number of axes: `"batch 2 *form"` is `(batch, 2, 384)`
from the flat loader and `(batch, 2, 16, 24)` from the grid.

Read it in atlas before you train anything: the shape on every arrow, and where a choice
leaves a stage out.

    atlas-code-nightly view notebooks/3_scaling_and_search/exercises

Run it from the root of the course repository to build every combination of the options
and see the shape after every stage:

    uv run notebooks/3_scaling_and_search/exercises/designs.py

To design your own:

1. Add a stage: a method of `Mix` (or of `Head`, for a block inside the head) whose
   signature says the shape it takes and the shape it gives, under
   `@jaxtyped(typechecker=beartype)`. Build its layers in `__init__`, only when the
   config chooses it.
2. If the search may leave it out, mark it `@option("<axis>")`, give `Config` a field
   `<axis>: bool = False`, and call it in `forward` under `if self.config.<axis>:`.
3. Add the axis to `SPACE` in `study.py`: `"<axis>": Choice([False, True])`.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from vectormesh.components import (
    Activation,
    Conv2D,
    Dense,
    Gate,
    NormKind,
    Pool2D,
    Projection,
    Serial,
    SparseMoE,
)
from vectormesh.design import Design, option

# the rows of the two vectors each input reads: 0 is the CLS token, 1 the patch mean
ROWS = {"cls": [0], "mean": [1], "both": [0, 1]}


@dataclass(frozen=True)
class Config:
    """The choices of the design. The search space draws them (`SPACE` in `study.py`);
    a field the space leaves out keeps its default here."""

    input: Literal["cls", "mean", "both"] = "cls"
    width: int = 64  # of every hidden layer
    depth: int = 1  # hidden layers
    norm: NormKind = "none"
    dropout: float = 0.0
    channels: int = 8  # filters of the kernel
    kernel_size: int = 3
    pool: Literal["max", "mean"] = "max"
    n_experts: int = 8
    kernel: bool = False
    skip: bool = False
    gate: bool = False
    moe: bool = False


class Head(nn.Module):
    """Features in, a score per class out: lesson 1's head. `depth` hidden layers of
    `width`, a skip around them and a gate on them, then the classifier."""

    skip_projection: Projection
    gating: Gate

    def __init__(self, config: Config, d_in: int, n_classes: int) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError(
                "Head: depth must be at least 1; without a hidden layer the head is the "
                "linear baseline, which study.py measures by itself"
            )
        self.config = config
        self.d_in = d_in
        self.width = config.width
        self.n_classes = n_classes
        # a Dense is Linear, Norm, GELU, Dropout; norm "none" and dropout 0 leave their
        # part in as the identity
        widths = [d_in] + [config.width] * config.depth
        self.block = Serial(
            [
                Dense(a, b, norm=config.norm, dropout=config.dropout)
                for a, b in pairwise(widths)
            ]
        )
        # an optional stage's layers exist only in the subgraph that uses them, so
        # n_params counts what the trial trained
        if config.skip:
            self.skip_projection = Projection(d_in, config.width)
        if config.gate:
            self.gating = Gate(config.width)
        self.classifier = Projection(config.width, n_classes)

    @jaxtyped(typechecker=beartype)
    def hidden(
        self, features: Float[Tensor, "batch {self.d_in}"]
    ) -> Float[Tensor, "batch {self.width}"]:
        return self.block(features)

    @option("skip")
    @jaxtyped(typechecker=beartype)
    def skip(
        self, features: Float[Tensor, "batch {self.d_in}"]
    ) -> Float[Tensor, "batch {self.width}"]:
        return self.skip_projection(features)

    @option("gate")
    @jaxtyped(typechecker=beartype)
    def gate(
        self, hidden: Float[Tensor, "batch {self.width}"]
    ) -> Float[Tensor, "batch {self.width}"]:
        return self.gating(hidden)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, hidden: Float[Tensor, "batch {self.width}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.classifier(hidden)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, features: Float[Tensor, "batch {self.d_in}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        hidden = self.hidden(features)
        if self.config.skip:
            hidden = hidden + self.skip(features)
        if self.config.gate:
            hidden = self.gate(hidden)
        return self.classify(hidden)


class Mix(nn.Module):
    """An image's two vectors in, a score per class out. Each stage is a method whose
    signature states the shape it takes and the shape it gives, in the widths this model
    was built with, and `@jaxtyped` checks that statement on every call."""

    kernels: Serial
    dense: Head
    experts: SparseMoE

    def __init__(
        self, config: Config, dim: int, height: int, breadth: int, n_classes: int
    ) -> None:
        super().__init__()
        if height * breadth != dim:
            raise ValueError(f"Mix: a grid of {height} x {breadth} does not hold {dim}")
        self.config = config
        self.dim, self.height, self.breadth = dim, height, breadth
        self.n_classes = n_classes
        self.rows = ROWS[config.input]
        self.n_in = len(self.rows)
        if config.kernel:
            # each chosen vector is one channel of the grid; the pool halves both sides
            k = config.kernel_size
            self.kernels = Serial(
                [
                    Conv2D(self.n_in, config.channels, k, padding=k // 2),
                    Activation("gelu"),
                    Pool2D(2, kind=config.pool),
                ]
            )
            self.d_in = config.channels * (height // 2) * (breadth // 2)
        else:
            self.d_in = self.n_in * dim
        if config.moe:
            # n_experts heads; the router reads the features and runs one head per image
            heads = [
                Head(config, self.d_in, n_classes) for _ in range(config.n_experts)
            ]
            self.experts = SparseMoE(heads, self.d_in, n_classes, k=1)
        else:
            self.dense = Head(config, self.d_in, n_classes)

    @jaxtyped(typechecker=beartype)
    def choose(
        self, image: Float[Tensor, "batch 2 *form"]
    ) -> Float[Tensor, "batch {self.n_in} *form"]:
        return image[:, self.rows]  # a vector or a grid per chosen row

    @jaxtyped(typechecker=beartype)
    def flatten(
        self, chosen: Float[Tensor, "batch {self.n_in} *form"]
    ) -> Float[Tensor, "batch {self.d_in}"]:
        return chosen.flatten(1)

    @option("kernel")
    @jaxtyped(typechecker=beartype)
    def kernel(
        self, grid: Float[Tensor, "batch {self.n_in} {self.height} {self.breadth}"]
    ) -> Float[Tensor, "batch {self.d_in}"]:
        return self.kernels(grid).flatten(1)  # (batch, channels, h/2, w/2), flattened

    @jaxtyped(typechecker=beartype)
    def head(
        self, features: Float[Tensor, "batch {self.d_in}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.dense(features)

    @option("moe")
    @jaxtyped(typechecker=beartype)
    def moe(
        self, features: Float[Tensor, "batch {self.d_in}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.experts(features)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, image: Float[Tensor, "batch 2 *form"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        chosen = self.choose(image)
        if self.config.kernel:
            if chosen.ndim != 4:
                raise ValueError(
                    "Mix: the kernel reads the grid the loader gives with form='grid'; "
                    f"this batch is {tuple(image.shape)}, put form='grid' in SPACE"
                )
            features = self.kernel(chosen)
        else:
            features = self.flatten(chosen)
        if self.config.moe:
            return self.moe(features)
        return self.head(features)


def control_width(flops: int, d_in: int, n_classes: int) -> int:
    """The width of the plain head (one hidden layer, every option off) that does `flops`
    FLOPs per image: the control of equal compute of a design that does `flops`. Such a
    head does `2 * (d_in * w + w * n_classes)` FLOPs per image."""
    return max(1, round(flops / (2 * (d_in + n_classes))))


# The model and the config that switches its options, as the study reads them; the head
# inside it, with its own options.
DESIGN = Design(Mix, Config)
HEAD = Design(Head, Config)


if __name__ == "__main__":
    # dinov2-small as the loader gives it with form="grid": two vectors of 384 numbers,
    # each folded 16 x 24; 321 classes. Every combination reads the grid (the plain head
    # flattens it); with form="flat", (2, 2, 384), the kernel is refused. Two images are
    # enough to trace every shape.
    DESIGN.show(
        torch.randn(2, 2, 16, 24), dim=384, height=16, breadth=24, n_classes=321
    )
    HEAD.show(torch.randn(2, 384), d_in=384, n_classes=321)
