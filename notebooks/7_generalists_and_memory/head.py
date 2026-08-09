"""The chunk-embedding pipeline shared by the head (13-way) and the router (2-way).

Both are the same architecture family over the same `(chunks, 1024)` SCOTUS embeddings --
`GaussianNoise -> Projection -> [TransformerBlock] -> Aggregator -> Dropout -> NeuralNet` --
differing only in `out_size` and in what the target column encodes. `03_who_can_the_head_serve`
searches this space to find the head; `04` reuses the winning config, with `k=2`, for the
router, on the grounds that a harder 13-way problem and an easier 2-way one over the same
embedding space are unlikely to want different architectures enough to justify a second search.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from vectormesh.components import (
    AttentionAggregator,
    FixedPadding,
    GaussianNoise,
    MaskedMeanAggregator,
    MeanAggregator,
    NeuralNet,
    Projection,
    RNNAggregator,
    Serial,
    TransformerBlock,
)
from vectormesh.components.metrics import Accuracy
from vectormesh.data import Collate
from vectormesh.training import Trainer, TrainerSettings

AGGREGATORS = {
    "mean": lambda d: MeanAggregator(),
    "masked_mean": lambda d: MaskedMeanAggregator(),
    "attention": lambda d: AttentionAggregator(hidden_size=d),
    "rnn": lambda d: RNNAggregator(hidden_size=d),
}


def build_pipeline(config: dict, d_in: int, k: int) -> Serial:
    parts = [
        GaussianNoise(sigma=config["sigma"], relative=True),
        Projection(hidden_size=d_in, out_size=config["hidden_dim"]),
    ]
    for _ in range(config["n_transformer_blocks"]):
        parts.append(
            TransformerBlock(
                hidden_size=config["hidden_dim"], num_heads=config["num_heads"]
            )
        )
    parts.append(AGGREGATORS[config["aggregator"]](config["hidden_dim"]))
    parts.append(torch.nn.Dropout(config["dropout"]))
    parts.append(NeuralNet(hidden_size=config["hidden_dim"], out_size=k))
    return Serial(parts)


def make_loader(
    ds,
    max_chunks: int,
    balanced: bool = False,
    shuffle: bool = True,
    batch_size: int = 32,
    label_col: str = "label",
) -> DataLoader:
    """`label_col` is what balancing balances *on* -- the 13-way SCOTUS label for the head,
    but a different column (e.g. the router's binary owned/tail target) for anything reusing
    this loader over a different classification target on the same cache."""
    collate = Collate(
        embedding_col="embed",
        target_col="onehot",
        padder=FixedPadding(max_chunks=max_chunks),
    )
    if balanced:
        labels = np.asarray([int(v) for v in ds[label_col]])
        freq = np.bincount(labels, minlength=int(labels.max()) + 1)
        weights = 1.0 / np.maximum(freq[labels], 1)
        sampler = WeightedRandomSampler(weights.tolist(), len(labels), replacement=True)
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate,
            num_workers=0,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0
    )


def predict(model, loader, device: str = "cpu"):
    """Argmax predictions and gold labels, in loader order."""
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X.to(device)).argmax(-1).cpu().tolist()
            gold += y.argmax(-1).tolist()
    return np.asarray(preds), np.asarray(gold)


def run(
    config: dict,
    train_ds,
    valid_ds,
    epochs: int,
    seed: int,
    d_in: int,
    k: int,
    earlystop_kwargs: dict | None = None,
    logdir: str = "screen",
    device: str = "cpu",
    label_col: str = "label",
):
    """Train one seed of `config` and return the model plus its predictions on `valid_ds`."""
    torch.manual_seed(seed)
    tr = make_loader(
        train_ds, config["max_chunks"], balanced=config["balanced"], label_col=label_col
    )
    va = make_loader(valid_ds, config["max_chunks"], shuffle=False)
    model = build_pipeline(config, d_in, k)
    settings = TrainerSettings(
        epochs=epochs,
        metrics=[Accuracy()],
        logdir=Path(f"ray_results/{logdir}").resolve(),
        train_steps=len(tr),
        valid_steps=len(va),
        earlystop_kwargs=earlystop_kwargs,
    )
    Trainer(
        model=model,
        settings=settings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=tr,
        validdataloader=va,
        device=device,
        progress=False,
    ).loop()
    pred, gold = predict(model, va, device)
    return model, pred, gold


def as_config(row) -> dict:
    """A ray dataframe row back to a plain config: its values are numpy scalars,
    which torch and json both refuse in their own ways."""
    return {
        k.replace("config/", ""): (v.item() if hasattr(v, "item") else v)
        for k, v in row.items()
        if k.startswith("config/")
    }
