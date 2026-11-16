"""The fusion arm of notebook 04 as a design: two encoders' vectors, a projection per
branch that the search may leave out, one head.

A tensor you want a shape on gets a name, and a name is a stage: a method whose signature
says the shape it takes and the shape it gives. Between two names the vectormesh
components compose as in the notebook (`Parallel`, a connector, a head). `forward` is the
whole design, one line per stage.

Run from the repository root to build both subgraphs and see the shape after every stage:

    uv run python notebooks/1_representation_and_head/fusion_architecture.py

Read it in atlas:

    atlas-code-nightly view notebooks/1_representation_and_head/fusion_architecture.py
"""

from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from vectormesh.components import Concatenate2D, NeuralNet, Parallel, Projection
from vectormesh.design import Design, option


@dataclass(frozen=True)
class Config:
    """The choices of the design."""

    shared_dim: int = 128  # each branch projects to this width before the join
    project: bool = True  # a projection per branch before the join


class Fusion(nn.Module):
    """Two streams, one per encoder, joined into one head. The model takes the pair a
    joined loader gives. Each stage is a method whose signature states the shape it takes
    and the shape it gives, in terms of the widths this model was built with, and
    `@jaxtyped` checks that statement on every call."""

    projections: Parallel

    def __init__(
        self, config: Config, dino_dim: int, resnet_dim: int, n_classes: int
    ) -> None:
        super().__init__()
        self.config = config
        self.dino_dim = dino_dim
        self.resnet_dim = resnet_dim
        self.shared_dim = config.shared_dim
        self.joined_dim = (
            2 * config.shared_dim if config.project else dino_dim + resnet_dim
        )
        self.n_classes = n_classes
        # an optional stage's layers exist only in the subgraph that uses them
        if config.project:
            self.projections = Parallel(
                [
                    Projection(dino_dim, config.shared_dim),
                    Projection(resnet_dim, config.shared_dim),
                ]
            )
        self.concatenate = Concatenate2D()
        self.head = NeuralNet(hidden_size=self.joined_dim, out_size=n_classes)

    @option("project")
    @jaxtyped(typechecker=beartype)
    def project(
        self,
        streams: tuple[
            Float[Tensor, "batch {self.dino_dim}"],
            Float[Tensor, "batch {self.resnet_dim}"],
        ],
    ) -> tuple[
        Float[Tensor, "batch {self.shared_dim}"],
        Float[Tensor, "batch {self.shared_dim}"],
    ]:
        return self.projections(streams)

    @jaxtyped(typechecker=beartype)
    def join(
        self, streams: tuple[Float[Tensor, "batch _"], Float[Tensor, "batch _"]]
    ) -> Float[Tensor, "batch {self.joined_dim}"]:
        return self.concatenate(streams)

    @jaxtyped(typechecker=beartype)
    def classify(
        self, joined: Float[Tensor, "batch {self.joined_dim}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.head(joined)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        streams: tuple[
            Float[Tensor, "batch {self.dino_dim}"],
            Float[Tensor, "batch {self.resnet_dim}"],
        ],
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        if self.config.project:
            streams = self.project(streams)
        joined = self.join(streams)
        return self.classify(joined)


# The model and the config that switches its option, as a search reads them.
DESIGN = Design(Fusion, Config)


def random_pair(
    batch_size: int, dino_dim: int, resnet_dim: int
) -> tuple[
    Float[Tensor, "{batch_size} {dino_dim}"], Float[Tensor, "{batch_size} {resnet_dim}"]
]:
    return torch.randn(batch_size, dino_dim), torch.randn(batch_size, resnet_dim)


def main() -> None:
    # dinov2-small is 384 wide, resnet-18 is 512, and DTD has 47 textures
    model = Fusion(Config(shared_dim=128), dino_dim=384, resnet_dim=512, n_classes=47)
    streams = random_pair(batch_size=64, dino_dim=384, resnet_dim=512)
    logits = model.forward(streams)
    print(model)
    print(
        f"{tuple(streams[0].shape)} + {tuple(streams[1].shape)} -> {tuple(logits.shape)}"
    )
    print()
    DESIGN.show(
        lambda chosen: Fusion(
            DESIGN.drawn(chosen), dino_dim=384, resnet_dim=512, n_classes=47
        ),
        random_pair(batch_size=2, dino_dim=384, resnet_dim=512),
    )


if __name__ == "__main__":
    main()
