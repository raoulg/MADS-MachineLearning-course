"""The architecture of a model as data: every layer with the shape it takes and gives,
and what one image costs it: the FLOPs of its forward pass and the time that pass takes.

A run carries it twice: as the trial's note, which optuna-dashboard renders, and as the
artifact `architecture.json`, which the class view reads.
"""

from __future__ import annotations

import statistics
import time

import jaxtyping
import torch
from pydantic import BaseModel
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary
from vectormesh.types import Padded, TensorInput

# the rows of the batch a forward pass is timed on, and how many passes are timed
LATENCY_ROWS = 256
LATENCY_CALLS = 20

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


def _parts(example: TensorInput) -> tuple[Tensor | Padded, ...]:
    return tuple(example) if isinstance(example, tuple | list) else (example,)


def _like(example: TensorInput, parts: tuple[Tensor | Padded, ...]) -> TensorInput:
    return parts if isinstance(example, tuple | list) else parts[0]


def _rows(example: TensorInput) -> int:
    return int(_parts(example)[0].shape[0])


def _on(example: TensorInput, device: torch.device) -> TensorInput:
    def moved(part: Tensor | Padded) -> Tensor | Padded:
        if isinstance(part, Padded):
            return Padded(values=part.values.to(device), mask=part.mask.to(device))
        return part.to(device)

    return _like(example, tuple(moved(part) for part in _parts(example)))


def _rows_like(part: Tensor, rows: int) -> Tensor:
    if part.is_floating_point():
        return torch.randn(rows, *part.shape[1:], dtype=part.dtype, device=part.device)
    return part[:1].expand(rows, *part.shape[1:]).clone()


def _batch_like(part: Tensor | Padded, rows: int) -> Tensor | Padded:
    """`rows` rows shaped as the rows of `part`: random for a float tensor, so a router
    spreads them over its experts as it would real rows; the first row repeated for any
    other input."""
    if isinstance(part, Padded):
        return Padded(
            values=_rows_like(part.values, rows), mask=_rows_like(part.mask, rows)
        )
    return _rows_like(part, rows)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def flops_per_image(model: nn.Module, example: TensorInput) -> int:
    """The floating-point operations of one forward pass of `model` over `example`,
    divided by its rows: a multiply and an add count as two. `FlopCounterMode` counts
    the operations that ran, so a sparse mixture of experts counts its router and the
    experts each row went to, not the experts it stores."""
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(example)
    return round(counter.get_total_flops() / _rows(example))


def latency_per_image(model: nn.Module, example: TensorInput) -> float:
    """Microseconds per image of one forward pass over `LATENCY_ROWS` rows shaped as
    those of `example`: the median of `LATENCY_CALLS` timed passes after five untimed
    ones, with torch on one CPU thread. The shape checks of jaxtyping are off while it times,
    as in a deployed model."""
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    batch = _like(
        example, tuple(_batch_like(part, LATENCY_ROWS) for part in _parts(example))
    )
    threads = torch.get_num_threads()
    checking = jaxtyping.config.jaxtyping_disable
    torch.set_num_threads(1)
    jaxtyping.config.update("jaxtyping_disable", True)
    times = []
    try:
        with torch.inference_mode():
            for _ in range(5):
                model(batch)
            for _ in range(LATENCY_CALLS):
                _sync(device)
                start = time.perf_counter()
                model(batch)
                _sync(device)
                times.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(threads)
        jaxtyping.config.update("jaxtyping_disable", checking)
    return statistics.median(times) * 1e6 / LATENCY_ROWS


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
    # the FLOPs of one forward pass, per image, and its time in microseconds per image
    flops_per_image: int | None = None
    latency_us: float | None = None
    input_shape: Shape | None
    output_shape: Shape | None
    layers: list[Layer]
    summary: str

    @classmethod
    def of(cls, model: nn.Module, example: TensorInput) -> Architecture:
        """The architecture of `model` on `example`, one batch input exactly as you would
        pass it as `model(example)`: a tensor, or a tuple of tensors for a model with
        two streams. Two rows are enough; they are moved to the model's device. The
        FLOPs are counted on `example` and the time on a batch of `LATENCY_ROWS` rows
        like it, both in eval mode; the model's mode is restored afterwards."""
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
        training = model.training
        try:
            model.eval()
            flops = flops_per_image(model, _on(example, device))
            latency = latency_per_image(model, _on(example, device))
        finally:
            model.train(training)
        stats = summary(
            model,
            input_data=[example],
            device=device,
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
            flops_per_image=flops,
            latency_us=round(latency, 3),
            input_shape=layers[0].input_shape if layers else None,
            output_shape=layers[0].output_shape if layers else None,
            layers=layers,
            summary=str(stats),
        )

    def note(self) -> str:
        """The markdown optuna-dashboard shows as the trial's note."""
        return f"`{self.signature}`\n\n```\n{self.summary}\n```"
