"""The first step of lesson 1 as a script: a VectorCache, cosine similarity, nearest neighbour, PCA.

Run from the repository root:

    uv run python notebooks/1_pytorch_intro/01_the_cache_demo.py
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float, Int
from loguru import logger
from vectormesh import VectorCache

REPO = "pttrn-io/oxford-flowers-dinov2-small"
PICTURE = Path(__file__).parent / "output" / "01_the_cache_pca.png"

# the library narrates every cache load; this script prints its own steps
logger.remove()


@dataclass(frozen=True)
class Vectors:
    """One vector per row: `values` has shape (count, dims), `labels` has shape (count,)."""

    values: Float[torch.Tensor, "items dim"]
    labels: Int[torch.Tensor, "items"]  # noqa: F821
    names: tuple[str, ...]

    @property
    def count(self) -> int:
        return self.values.shape[0]

    @property
    def dims(self) -> int:
        return self.values.shape[1]

    def __repr__(self) -> str:
        return f"Vectors({self.count} x {self.dims} dims, {len(self.names)} classes)"


def load_vectors(split: str) -> Vectors:
    cache = VectorCache.from_hub(REPO, split=split)
    return Vectors(
        values=cache["embed"][:],
        labels=cache["label"][:],
        names=tuple(cache.features["label"].names),
    )


def cosine_similarity(
    queries: Float[torch.Tensor, "queries dim"],
    keys: Float[torch.Tensor, "keys dim"],
) -> Float[torch.Tensor, "queries keys"]:
    """Shape (q, dims) and (k, dims) to (q, k), each entry between -1 and 1.

    Scale every vector to length 1 first. Then the dot product of two vectors is the cosine
    of the angle between them: 1 means the same direction, 0 means unrelated, -1 opposite.
    """
    unit_queries = torch.nn.functional.normalize(queries, dim=1)
    unit_keys = torch.nn.functional.normalize(keys, dim=1)
    return unit_queries @ unit_keys.T


class OneNearestNeighbour:
    """No training: the model is the stored vectors. A prediction copies the label of the closest one."""

    def __init__(self, train: Vectors) -> None:
        self.train = train

    def predict(self, test: Vectors) -> Int[torch.Tensor, "queries"]:  # noqa: F821
        similarity = cosine_similarity(test.values, self.train.values)
        neighbour = similarity.argmax(dim=1)
        return self.train.labels[neighbour]


def accuracy(
    predicted: Int[torch.Tensor, "items"],  # noqa: F821
    truth: Int[torch.Tensor, "items"],  # noqa: F821
) -> float:
    return (predicted == truth).float().mean().item()


def fit_pca(
    values: Float[torch.Tensor, "items dim"], components: int = 2
) -> tuple[Float[torch.Tensor, "dim"], Float[torch.Tensor, "dim components"]]:  # noqa: F821
    """The mean, shape (dims,), and the directions of largest spread, shape (dims, components)."""
    mean = values.mean(dim=0)
    _, _, basis = torch.pca_lowrank(values - mean, q=components)
    return mean, basis


def project(
    values: Float[torch.Tensor, "items dim"],
    mean: Float[torch.Tensor, "dim"],  # noqa: F821
    basis: Float[torch.Tensor, "dim components"],
) -> Float[torch.Tensor, "items components"]:
    """Shape (count, dims) to (count, components): where each vector lands on those directions."""
    return (values - mean) @ basis


def pick_classes(vectors: Vectors, count: int = 8, seed: int = 0) -> Vectors:
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(len(vectors.names), generator=generator)[:count]
    mask = torch.isin(vectors.labels, chosen)
    return Vectors(vectors.values[mask], vectors.labels[mask], vectors.names)


def plot_pca(
    points: Float[torch.Tensor, "items components"],
    labels: Int[torch.Tensor, "items"],  # noqa: F821
    names: tuple[str, ...],
    dims: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in labels.unique():
        mask = labels == label
        ax.scatter(
            points[mask, 0], points[mask, 1], s=14, alpha=0.8, label=names[int(label)]
        )
    ax.set_title(f"{dims} dims projected onto 2 principal components")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    train = load_vectors("train")
    test = load_vectors("test")
    print(f"train   {train}")
    print(f"test    {test}")

    predicted = OneNearestNeighbour(train).predict(test)
    print(f"1-nearest-neighbour accuracy: {accuracy(predicted, test.labels):.4f}")
    print(f"chance level:                 {1 / len(train.names):.4f}")

    mean, basis = fit_pca(train.values)
    subset = pick_classes(test)
    points = project(subset.values, mean, basis)
    plot_pca(points, subset.labels, subset.names, train.dims, PICTURE)
    print(f"picture: {PICTURE}")


if __name__ == "__main__":
    main()
