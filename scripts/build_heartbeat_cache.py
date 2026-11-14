"""Build the ECG heartbeat VectorCaches, one per (dataset x split), ready to publish.

    uv run python scripts/build_heartbeat_cache.py

Reads the four parquet files of raoulg/mads-exam-24, cloned into
artefacts/sources/mads-exam-24 (`make build-heartbeats` clones it when it is missing):

    heart_train.parq, heart_test.parq            PTB Diagnostic, 2 classes
    heart_big_train.parq, heart_big_test.parq    MIT-BIH Arrhythmia, 5 classes

Every row is one heartbeat: 187 steps at 125 Hz, zero-padded at the end, and a `target`
column. There is no encoder: the vectorizer below hands the 187 steps over unchanged as a
float32 `(187, 1)` tensor (sequence, channel), so the cache is rank 2 and a DataLoader
over it gives `(batch, 187, 1)`. Its model_name is `raw`, which names the repos
`pttrn-io/heartbeats-ptb-raw` and `pttrn-io/heartbeats-mitbih-raw`.

Kept columns: the vector, `label` (int, from `target`) and `source_idx`, the row's
position in its source parquet file. Nothing is shuffled or subsampled. After each
build, every cached beat and label is compared with its source row by `source_idx`, and
a mismatch stops the script.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
from beartype import beartype
from datasets import Dataset
from jaxtyping import Float, jaxtyped
from pydantic import model_validator
from torch import Tensor
from vectormesh import VectorCache
from vectormesh.data.vectorizers import ChunkedVectorizer

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ROOT / "artefacts"
SOURCE = ARTEFACTS / "sources" / "mads-exam-24" / "data"

STEPS = 187
INPUT_COL = "steps"

# dataset tag -> {split: parquet file}
DATASETS = {
    "heartbeats-ptb": {"train": "heart_train.parq", "test": "heart_test.parq"},
    "heartbeats-mitbih": {
        "train": "heart_big_train.parq",
        "test": "heart_big_test.parq",
    },
}


class RawSequenceVectorizer(ChunkedVectorizer):
    """Hand each row's steps over unchanged, as a float32 `(steps, 1)` tensor.

    One channel per step makes the cache rank 2, `(sequence, channel)`, the shape a
    sequence model takes, rather than a flat `(187,)` feature vector.
    """

    model_name: str = "raw"
    col_name: str = "beat"
    input_col: str = INPUT_COL

    @model_validator(mode="after")
    def initialize_model(self) -> Self:
        class RawMetadata:
            hidden_size = 1  # one channel

        self._metadata = RawMetadata()
        self._effective_max_length = None
        return self

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, inputs: list, batchsize: int
    ) -> dict[str, list[Float[Tensor, "_ dim"]]]:
        batch = torch.tensor(np.asarray(inputs, dtype=np.float32)).unsqueeze(-1)
        return {self.col_name: list(batch)}


def read_split(path: Path) -> tuple[Dataset, np.ndarray, np.ndarray]:
    """The parquet file as a Dataset of `steps`, `label` and `source_idx`, in file order.

    Also returns the steps `(rows, 187)` and labels `(rows,)` as arrays, to check the
    built cache against.
    """
    if not path.exists():
        sys.exit(
            f"{path} not found; clone the source first:\n"
            "  gh repo clone raoulg/mads-exam-24 artefacts/sources/mads-exam-24"
        )
    df = pd.read_parquet(path)
    steps = df.drop(columns="target").to_numpy(dtype=np.float32)
    if steps.shape[1] != STEPS:
        sys.exit(f"{path} has {steps.shape[1]} step columns, expected {STEPS}")
    # The PTB files carry a shuffled pandas index; source_idx is the position, which is
    # what `pd.read_parquet(path).iloc[source_idx]` resolves.
    labels = df["target"].astype(int).to_numpy()
    dataset = Dataset.from_dict(
        {
            INPUT_COL: list(steps),
            "label": labels.tolist(),
            "source_idx": list(range(len(df))),
        }
    )
    return dataset, steps, labels


def check_cache(folder: Path, steps: np.ndarray, labels: np.ndarray) -> None:
    """Exit unless every beat and label in the cache at `folder` equals its source row."""
    # Read the Arrow columns directly: decoding the nested beats row by row through
    # the numpy or torch format takes minutes on MIT-BIH train.
    table = VectorCache.load(folder).dataset.data.table
    idx = table["source_idx"].to_numpy()
    rows = table["beat"].combine_chunks()
    steps_per_row = rows.value_lengths().to_numpy()
    channels = rows.flatten()
    channels_per_step = channels.value_lengths().to_numpy()
    problems = []
    if len(rows) != len(steps):
        problems.append(f"{len(rows)} rows, source has {len(steps)}")
    elif not np.array_equal(np.sort(idx), np.arange(len(steps))):
        problems.append("source_idx is not a permutation of the source rows")
    else:
        beats = channels.flatten().to_numpy().reshape(-1, STEPS, 1)
        if (steps_per_row != STEPS).any() or (channels_per_step != 1).any():
            problems.append("not every beat is (187, 1)")
        elif beats.dtype != np.float32:
            problems.append(f"beats are {beats.dtype}")
        elif not np.array_equal(beats[:, :, 0], steps[idx]):
            bad = np.flatnonzero((beats[:, :, 0] != steps[idx]).any(axis=1))
            problems.append(
                f"{len(bad)} beats differ from the source, first row {bad[0]}"
            )
        if not np.array_equal(table["label"].to_numpy(), labels[idx]):
            problems.append("labels differ from the source")
    if problems:
        sys.exit(f"{folder.name}: " + "; ".join(problems))
    print(f"  checked: all {len(steps)} beats and labels equal their source rows")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help="build only this dataset (repeatable); default: both",
    )
    p.add_argument("--map-batch", type=int, default=1024)
    args = p.parse_args()

    vectorizer = RawSequenceVectorizer()
    for tag in args.dataset or DATASETS:
        for split, filename in DATASETS[tag].items():
            dataset, steps, labels = read_split(SOURCE / filename)
            name = f"{tag}_{split}_{vectorizer.model_name}"
            print(f"building {name}: {len(dataset)} rows from {filename}")
            start = time.time()
            cache = VectorCache.create(
                cache_dir=ARTEFACTS,
                vectorizer=vectorizer,
                dataset=dataset,
                dataset_tag=name,
                remove_columns=[INPUT_COL],
                map_batch=args.map_batch,
            )
            print(
                f"  {cache.name}  ({time.time() - start:.0f}s)\n"
                f"  columns: {cache.column_names}"
            )
            check_cache(ARTEFACTS / cache.name, steps, labels)

    print("\nnow check the cards:\n  make publish-heartbeats ARGS=--dry-run")


if __name__ == "__main__":
    main()
