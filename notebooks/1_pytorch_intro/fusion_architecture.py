"""The fusion arm of notebook 04 as one architecture: two encoders' vectors, two branches, one head.

Run from the repository root:

    uv run python notebooks/1_pytorch_intro/fusion_architecture.py
"""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from vectormesh.components import Concatenate2D, NeuralNet, Projection


class Fusion(nn.Module):
    """Two branches, one per encoder, joined into one head. Each stage is a method whose
    signature states the shape it takes and the shape it gives, in terms of the widths
    this model was built with, and `@jaxtyped` checks that statement on every call."""

    def __init__(
        self, dino_dim: int, resnet_dim: int, shared_dim: int, n_classes: int
    ) -> None:
        super().__init__()
        self.dino_dim = dino_dim
        self.resnet_dim = resnet_dim
        self.shared_dim = shared_dim
        self.joined_dim = 2 * shared_dim
        self.n_classes = n_classes
        self.dino_projection = Projection(dino_dim, shared_dim)
        self.resnet_projection = Projection(resnet_dim, shared_dim)
        self.concatenate = Concatenate2D()
        self.head = NeuralNet(hidden_size=self.joined_dim, out_size=n_classes)

    @jaxtyped(typechecker=beartype)
    def dino_branch(
        self, dino: Float[Tensor, "batch {self.dino_dim}"]
    ) -> Float[Tensor, "batch {self.shared_dim}"]:
        return self.dino_projection(dino)

    @jaxtyped(typechecker=beartype)
    def resnet_branch(
        self, resnet: Float[Tensor, "batch {self.resnet_dim}"]
    ) -> Float[Tensor, "batch {self.shared_dim}"]:
        return self.resnet_projection(resnet)

    @jaxtyped(typechecker=beartype)
    def join(
        self,
        dino_z: Float[Tensor, "batch {self.shared_dim}"],
        resnet_z: Float[Tensor, "batch {self.shared_dim}"],
    ) -> Float[Tensor, "batch {self.joined_dim}"]:
        return self.concatenate((dino_z, resnet_z))

    @jaxtyped(typechecker=beartype)
    def classify(
        self, joined: Float[Tensor, "batch {self.joined_dim}"]
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        return self.head(joined)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        dino: Float[Tensor, "batch {self.dino_dim}"],
        resnet: Float[Tensor, "batch {self.resnet_dim}"],
    ) -> Float[Tensor, "batch {self.n_classes}"]:
        dino_z = self.dino_branch(dino)
        resnet_z = self.resnet_branch(resnet)
        joined = self.join(dino_z, resnet_z)
        return self.classify(joined)


@jaxtyped(typechecker=beartype)
def random_batch(
    batch_size: int, dino_dim: int, resnet_dim: int
) -> tuple[
    Float[Tensor, "{batch_size} {dino_dim}"], Float[Tensor, "{batch_size} {resnet_dim}"]
]:
    return torch.randn(batch_size, dino_dim), torch.randn(batch_size, resnet_dim)


def main() -> None:
    model = Fusion(
        dino_dim=384,  # dinov2-small
        resnet_dim=512,  # resnet-18
        shared_dim=128,  # each branch projects to this width before the join
        n_classes=47,  # DTD textures
    )
    dino, resnet = random_batch(
        batch_size=64, dino_dim=model.dino_dim, resnet_dim=model.resnet_dim
    )
    logits = model.forward(dino, resnet)
    print(model)
    print(f"{tuple(dino.shape)} + {tuple(resnet.shape)} -> {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
