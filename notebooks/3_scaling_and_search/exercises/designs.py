"""The design of your study: lesson 2's model of one heartbeat, with a route stage and its
capacity-matched control as options.

A beat is `(batch, 187, 1)`: 187 steps, one channel. The design reads it with kernels,
shortens it block by block, and classifies it as N, S or V. Every block the search may
leave out is a choice in `forward` (`if self.config.route:`), and `@option("route")` on
its stage names the axis of the search space that decides it. The config a trial draws
picks the subgraph that is built. `option` and `Design` are in `vectormesh.design`, the
same for every lesson.

The choices of lesson 2's design, each a bet about the data:

- `kernel_size`, `width`, `depth`: how far one position sees, through depth; `width`
  and `depth` are also the parameter axis of a capacity curve;
- `downsample`: a pool of 2 or a kernel with stride 2 after every block;
- `parallel`, `fold`, `skip`, `excite`, `state`, `gate`: the blocks of lesson 2;
- `sigma`, `norm`, `dropout`: input noise, a norm in every block, dropout before the head.

What lesson 3 adds (notebook 3.1):

- `route`: a mixture of `n_experts` experts on every position of the shortened beat
  (`self.short`: 93, 46, 23 and 11 positions at `depth` 1 to 4, with a pool), each
  expert a `width -> width` layer with a GELU, and a router that weighs them per
  position. `routing="dense"` blends every expert with the router's softmax weights
  (`MoE`); `routing="sparse"` runs only the expert with the largest weight and scales
  its output by that weight (`SparseMoE`, top-1), so it trains all experts and uses one
  per position.
- `matched`: the control of `route`. With `route` on, `matched` puts a plain stage of the
  same parameter count in its place: two linear layers with a GELU between them, no
  router and no experts. The logged `n_params` of the two shows they are matched. Without
  `route`, `matched` changes nothing. `matched` matches the trained parameters, not the
  ones used per position: with `routing="dense"` both use all of them at every position
  (and about the same compute), but with `routing="sparse"` the route stage runs one
  expert per position, so `matched` uses more parameters, and more compute, per
  position than the route stage it stands in for.
- `balance`: the weight of the load-balancing loss (lesson 3 slides), 0 by default.
  The trainer adds `balance * load_balancing_loss(model)` to the loss of every training
  step and never to the validation loss. The loss is 1 when the router spreads the
  positions evenly over the experts and grows as it concentrates them on few, so a
  weight above 0 pushes against router collapse. Only `SparseMoE` computes it: with
  `routing="dense"` the weight changes nothing, and the trainer warns. Fedus et al.
  (2021) use 0.01.

A route stage and its control differ in the router and in the form of the function, at
the same parameter count: `n_experts` experts of `width -> width`, each ending in a GELU
and weighed per position, against one path `width -> hidden -> width` (hidden about twice
the width) whose last layer is linear and feeds the linear head. That is what a
capacity-matched control measures: whether weighing experts per position buys more than
the same parameters spent on a plain path.

Read it in atlas before you train anything: the shape on every arrow, and where a choice
leaves a stage out.

    atlas-code-nightly view notebooks/3_scaling_and_search/exercises

Run it from the root of the course repository to build every combination of the options
and see the shape after every stage:

    uv run notebooks/3_scaling_and_search/exercises/designs.py

To design your own:

1. Add a stage: a method of `Beats` whose signature says the shape it takes and the
   shape it gives, under `@jaxtyped(typechecker=beartype)`. Build its layers in
   `__init__`, only when the config chooses it.
2. If the search may leave it out, mark it `@option("<axis>")`, give `Config` a field
   `<axis>: bool = False`, and call it in `forward` under `if self.config.<axis>:`.
3. Add the axis to `SPACE` in `study.py`: `"<axis>": Choice([False, True])`.
4. A stage that adds parameters needs a control of the same parameter count, as
   `matched` is for `route`; without one, a gain says only that more parameters helped.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from vectormesh.components import (
    Conv1D,
    Conv2D,
    Gate,
    GaussianNoise,
    MoE,
    Norm,
    Pool1D,
    Recurrent,
    SparseMoE,
)
from vectormesh.design import Design, option

STEPS = 187
FOLD = (11, 17)  # 11 * 17 = 187: the beat as a grid of 11 rows of 17 steps


@dataclass(frozen=True)
class Config:
    """The choices of the design. The search space draws them (`SPACE` in `study.py`);
    a field the space leaves out keeps its default here."""

    kernel_size: int = 5
    width: int = 32  # channels after the first kernel
    depth: int = 3  # blocks, each followed by a downsample of 2
    downsample: Literal["pool", "stride"] = "pool"
    norm: Literal["none", "batch", "layer"] = "none"
    sigma: float = 0.0  # Gaussian noise on the beat, relative to its spread
    dropout: float = 0.0  # before the head
    wide_kernel: int = 15  # the second kernel of `parallel`
    n_experts: int = 4  # the experts of `route`
    routing: Literal["dense", "sparse"] = "dense"
    balance: float = 0.0  # the weight of the load-balancing loss in training
    parallel: bool = False
    fold: bool = False
    skip: bool = False
    excite: bool = False
    state: bool = False
    route: bool = False
    matched: bool = False
    gate: bool = False


def shortened(steps: int, depth: int, downsample: str) -> int:
    """The length of the sequence after `depth` downsamples of 2."""
    for _ in range(depth):
        steps = steps // 2 if downsample == "pool" else (steps + 1) // 2
    return steps


def route_params(width: int, n_experts: int) -> int:
    """The parameters of the route stage: `n_experts` experts of `width -> width` and a
    router of `width -> n_experts`, together `n_experts * (width + 1) ** 2`."""
    return n_experts * (width + 1) ** 2


def matched_hidden(width: int, n_experts: int) -> int:
    """The hidden size of the plain stage `width -> hidden -> width` whose parameters,
    `hidden * (2 * width + 1) + width`, come closest to the route stage's."""
    return max(1, round((route_params(width, n_experts) - width) / (2 * width + 1)))


class SqueezeExcite(nn.Module):
    """A gate on whole channels: squeeze the sequence axis away, then a narrow network
    and a sigmoid give every channel one multiplier between 0 and 1."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        narrow = max(channels // reduction, 1)
        self.excite = nn.Sequential(
            nn.Linear(channels, narrow),
            nn.GELU(),
            nn.Linear(narrow, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.excite(x.mean(dim=1)).unsqueeze(1)


class Beats(nn.Module):
    """A beat in, a score per class out. Each stage is a method whose signature states the
    shape it takes and the shape it gives, in the widths this model was built with, and
    `@jaxtyped` checks that statement on every call."""

    wide: Conv1D
    join: Conv1D
    grid: Conv2D
    excitations: nn.ModuleList
    recurrent: Recurrent
    experts: MoE | SparseMoE
    plain: nn.Sequential
    gating: Gate

    def __init__(self, config: Config, n_classes: int) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError(
                "Beats: depth must be at least 1; without a block the model is the "
                "linear baseline, which study.py measures by itself"
            )
        self.config = config
        self.width = config.width
        self.n_classes = n_classes
        self.steps = STEPS
        self.short = shortened(STEPS, config.depth, config.downsample)
        width, k = config.width, config.kernel_size

        self.noise = GaussianNoise(sigma=config.sigma)
        self.narrow = Conv1D(1, width, k, padding=k // 2)
        # an optional stage's layers exist only in the subgraph that uses them, so
        # n_params counts what the trial trained
        if config.parallel:
            wk = config.wide_kernel
            self.wide = Conv1D(1, width, wk, padding=wk // 2)
            self.join = Conv1D(2 * width, width, 1)
        if config.fold:
            self.grid = Conv2D(1, width, 3, padding=1)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for _ in range(config.depth):
            layers: list[nn.Module] = [Conv1D(width, width, k, padding=k // 2)]
            if config.norm != "none":
                layers.append(Norm(width, kind=config.norm))
            layers.append(nn.GELU())
            self.blocks.append(nn.Sequential(*layers))
            self.downs.append(
                Pool1D(2)
                if config.downsample == "pool"
                else Conv1D(width, width, 3, stride=2, padding=1)
            )
        if config.excite:
            self.excitations = nn.ModuleList(
                SqueezeExcite(width) for _ in range(config.depth)
            )
        if config.state:
            self.recurrent = Recurrent(width, width, kind="gru")
        if config.gate:
            self.gating = Gate(width)
        self.drop = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.short * width, n_classes)
        # the route stage or its control is built last: from one seed, the two arms of a
        # pair then start from the same weights in every other layer, the head included
        if config.route and config.matched:
            hidden = matched_hidden(width, config.n_experts)
            self.plain = nn.Sequential(
                nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)
            )
        elif config.route:
            experts: list[nn.Module] = [
                nn.Sequential(nn.Linear(width, width), nn.GELU())
                for _ in range(config.n_experts)
            ]
            self.experts = (
                MoE(experts, hidden_size=width, out_size=width)
                if config.routing == "dense"
                else SparseMoE(experts, hidden_size=width, out_size=width, k=1)
            )

    @jaxtyped(typechecker=beartype)
    def perturb(
        self, beat: Float[Tensor, "batch {self.steps} 1"]
    ) -> Float[Tensor, "batch {self.steps} 1"]:
        return self.noise(beat)

    @jaxtyped(typechecker=beartype)
    def kernel(
        self, beat: Float[Tensor, "batch {self.steps} 1"]
    ) -> Float[Tensor, "batch {self.steps} {self.width}"]:
        return self.narrow(beat)

    @option("parallel")
    @jaxtyped(typechecker=beartype)
    def parallel(
        self,
        beat: Float[Tensor, "batch {self.steps} 1"],
        narrow: Float[Tensor, "batch {self.steps} {self.width}"],
    ) -> Float[Tensor, "batch {self.steps} {self.width}"]:
        both = torch.cat([narrow, self.wide(beat)], dim=-1)  # (batch, steps, 2 width)
        return self.join(both)

    @option("fold")
    @jaxtyped(typechecker=beartype)
    def fold(
        self, beat: Float[Tensor, "batch {self.steps} 1"]
    ) -> Float[Tensor, "batch {self.steps} {self.width}"]:
        grid = beat.reshape(-1, 1, *FOLD)  # (batch, 1, 11, 17)
        seen = self.grid(grid)  # (batch, width, 11, 17)
        return seen.flatten(2).transpose(1, 2)  # back to (batch, 187, width)

    @jaxtyped(typechecker=beartype)
    def block(
        self, features: Float[Tensor, "batch steps {self.width}"], i: int
    ) -> Float[Tensor, "batch steps {self.width}"]:
        return self.blocks[i](features)

    @option("skip")
    @jaxtyped(typechecker=beartype)
    def skip(
        self,
        features: Float[Tensor, "batch steps {self.width}"],
        block: Float[Tensor, "batch steps {self.width}"],
    ) -> Float[Tensor, "batch steps {self.width}"]:
        return features + block

    @option("excite")
    @jaxtyped(typechecker=beartype)
    def excite(
        self, features: Float[Tensor, "batch steps {self.width}"], i: int
    ) -> Float[Tensor, "batch steps {self.width}"]:
        return self.excitations[i](features)

    @jaxtyped(typechecker=beartype)
    def downsample(
        self, features: Float[Tensor, "batch steps {self.width}"], i: int
    ) -> Float[Tensor, "batch shorter {self.width}"]:
        return self.downs[i](features)

    @option("state")
    @jaxtyped(typechecker=beartype)
    def state(
        self, features: Float[Tensor, "batch {self.short} {self.width}"]
    ) -> Float[Tensor, "batch {self.short} {self.width}"]:
        return self.recurrent(features)

    @option("route")
    @jaxtyped(typechecker=beartype)
    def route(
        self, features: Float[Tensor, "batch {self.short} {self.width}"]
    ) -> Float[Tensor, "batch {self.short} {self.width}"]:
        # the router reads every position on its own and weighs (dense) or picks
        # (sparse) the experts for it; the shape stays as it is
        return self.experts(features)

    @option("matched")
    @jaxtyped(typechecker=beartype)
    def matched(
        self, features: Float[Tensor, "batch {self.short} {self.width}"]
    ) -> Float[Tensor, "batch {self.short} {self.width}"]:
        # the route stage's parameters spent on one plain path, the same at every position
        return self.plain(features)

    @option("gate")
    @jaxtyped(typechecker=beartype)
    def gate(
        self, features: Float[Tensor, "batch {self.short} {self.width}"]
    ) -> Float[Tensor, "batch {self.short} {self.width}"]:
        return self.gating(features)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, features: Float[Tensor, "batch {self.short} {self.width}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.classifier(self.drop(features.flatten(1)))

    @jaxtyped(typechecker=beartype)
    def forward(
        self, beat: Float[Tensor, "batch {self.steps} 1"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        beat = self.perturb(beat)
        features = self.kernel(beat)
        if self.config.parallel:
            features = self.parallel(beat, features)
        if self.config.fold:
            features = features + self.fold(beat)
        for i in range(self.config.depth):
            block = self.block(features, i)
            if self.config.skip:
                block = self.skip(features, block)
            if self.config.excite:
                block = self.excite(block, i)
            features = self.downsample(block, i)
        if self.config.state:
            features = self.state(features)
        if self.config.route and self.config.matched:
            features = self.matched(features)
        elif self.config.route:
            features = self.route(features)
        if self.config.gate:
            features = self.gate(features)
        return self.classify(features)


# The model and the config that switches its options, as the study reads them.
DESIGN = Design(Beats, Config)


if __name__ == "__main__":
    # two beats are enough to trace every shape
    DESIGN.show(torch.randn(2, STEPS, 1), n_classes=3)
    # the route stage against its control, at the default width and n_experts
    for matched in (False, True):
        model = DESIGN.build({"route": True, "matched": matched}, n_classes=3)
        stage = model.plain if matched else model.experts
        n_params = sum(p.numel() for p in stage.parameters())
        print(f"{'matched' if matched else 'route':<8} stage: {n_params:,} params")
