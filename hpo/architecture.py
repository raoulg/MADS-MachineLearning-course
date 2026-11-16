"""The architecture of a model as data: every layer with the shape it takes and gives.

A run carries it twice: as the trial's note, which optuna-dashboard renders, and as the
artifact `architecture.json`, which the class view reads.
"""

from __future__ import annotations

from pydantic import BaseModel
from torch import nn
from torchinfo import summary
from vectormesh.types import TensorInput

# the shape of one input, or of each input of a model with more than one stream
Shape = list[int] | list[list[int]]


def _shape(size: object) -> Shape | None:
    if isinstance(size, list | tuple) and all(isinstance(d, int) for d in size):
        return [int(d) for d in size]
    if isinstance(size, list | tuple) and all(
        isinstance(s, list | tuple) and all(isinstance(d, int) for d in s) for s in size
    ):
        return [[int(d) for d in s] for s in size]
    return None


def _last_dim(shape: Shape | None) -> int | None:
    return shape[-1] if shape and isinstance(shape[-1], int) else None


class Layer(BaseModel):
    depth: int
    type: str
    name: str
    input_shape: Shape | None
    output_shape: Shape | None
    params: int | None  # for a leaf layer: the layers inside a block count for it
    leaf: bool

    def step(self) -> str:
        """The layer in the one-line signature: `Linear(384->256)`, or its type."""
        d_in, d_out = _last_dim(self.input_shape), _last_dim(self.output_shape)
        if self.params and d_in is not None and d_out is not None:
            return f"{self.type}({d_in}->{d_out})"
        return self.type


class Architecture(BaseModel):
    signature: str
    n_params: int
    n_trainable: int
    input_shape: Shape | None
    output_shape: Shape | None
    layers: list[Layer]
    summary: str

    @classmethod
    def of(cls, model: nn.Module, example: TensorInput) -> Architecture:
        """The architecture of `model` on `example`, one batch input exactly as you would
        pass it as `model(example)`: a tensor, or a tuple of tensors for a model with
        two streams. Two rows are enough."""
        stats = summary(
            model,
            input_data=[example],
            verbose=0,
            depth=8,
            mode="eval",
            col_names=["input_size", "output_size", "num_params"],
        )
        layers = [
            Layer(
                depth=info.depth,
                type=info.class_name,
                name=info.var_name,
                input_shape=_shape(info.input_size),
                output_shape=_shape(info.output_size),
                params=info.num_params if info.is_leaf_layer else None,
                leaf=info.is_leaf_layer,
            )
            for info in stats.summary_list
            if info.class_name != "ModuleList"
        ]
        return cls(
            signature=" > ".join(layer.step() for layer in layers if layer.leaf),
            n_params=int(stats.total_params),
            n_trainable=int(stats.trainable_params),
            input_shape=layers[0].input_shape if layers else None,
            output_shape=layers[0].output_shape if layers else None,
            layers=layers,
            summary=str(stats),
        )

    def note(self) -> str:
        """The markdown optuna-dashboard shows as the trial's note."""
        return f"`{self.signature}`\n\n```\n{self.summary}\n```"
