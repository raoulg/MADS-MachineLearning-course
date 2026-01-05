from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image
from torchvision.datasets import CIFAR10


CIFAR10_LABELS = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path = Path("data/raw")
    out_dir: Path = Path("data/processed/cifar3")

    classes: tuple[str, str, str] = ("cat", "dog", "horse")
    seed: int = 42

    # Bewust klein voor CPU-hypertuning
    n_train_per_class: int = 800
    n_val_per_class: int = 160
    n_test_per_class: int = 160


def ensure_dirs(base: Path, classes: tuple[str, ...]) -> None:
    for split in ("train", "val", "test"):
        for cls in classes:
            (base / split / cls).mkdir(parents=True, exist_ok=True)


def save_images(
    images,  # numpy array (H,W,C)
    indices: list[int],
    out_dir: Path,
    prefix: str,
) -> None:
    for j, idx in enumerate(indices):
        img = images[idx]
        Image.fromarray(img).save(out_dir / f"{prefix}_{j:04d}.png")


def main() -> None:
    cfg = DataConfig()
    random.seed(cfg.seed)

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(cfg.out_dir, cfg.classes)

    class_to_idx = {name: i for i, name in enumerate(CIFAR10_LABELS)}
    wanted_idxs = [class_to_idx[c] for c in cfg.classes]

    # Download datasets (train + test)
    train_ds = CIFAR10(root=str(cfg.raw_dir), train=True, download=True)
    test_ds = CIFAR10(root=str(cfg.raw_dir), train=False, download=True)

    # --- TRAIN/VAL: neem uit officiële train-set
    per_class_train: dict[int, list[int]] = {c: [] for c in wanted_idxs}
    for i, y in enumerate(train_ds.targets):
        if y in per_class_train:
            per_class_train[y].append(i)

    for cls_name in cfg.classes:
        cls_idx = class_to_idx[cls_name]
        idxs = per_class_train[cls_idx]
        random.shuffle(idxs)

        need = cfg.n_train_per_class + cfg.n_val_per_class
        if len(idxs) < need:
            raise ValueError(
                f"Not enough samples for class '{cls_name}'. "
                f"Needed {need}, found {len(idxs)}."
            )

        picked = idxs[:need]
        train_idxs = picked[: cfg.n_train_per_class]
        val_idxs = picked[cfg.n_train_per_class :]

        save_images(
            images=train_ds.data,
            indices=train_idxs,
            out_dir=cfg.out_dir / "train" / cls_name,
            prefix=cls_name,
        )
        save_images(
            images=train_ds.data,
            indices=val_idxs,
            out_dir=cfg.out_dir / "val" / cls_name,
            prefix=cls_name,
        )

    # --- TEST: neem uit officiële test-set
    per_class_test: dict[int, list[int]] = {c: [] for c in wanted_idxs}
    for i, y in enumerate(test_ds.targets):
        if y in per_class_test:
            per_class_test[y].append(i)

    for cls_name in cfg.classes:
        cls_idx = class_to_idx[cls_name]
        idxs = per_class_test[cls_idx]
        random.shuffle(idxs)

        if len(idxs) < cfg.n_test_per_class:
            raise ValueError(
                f"Not enough test samples for class '{cls_name}'. "
                f"Needed {cfg.n_test_per_class}, found {len(idxs)}."
            )

        picked = idxs[: cfg.n_test_per_class]
        save_images(
            images=test_ds.data,
            indices=picked,
            out_dir=cfg.out_dir / "test" / cls_name,
            prefix=cls_name,
        )

    # --- Metadata voor reproduceerbaarheid
    meta = cfg.out_dir / "dataset_meta.txt"
    meta.write_text(
        "\n".join(
            [
                "dataset: CIFAR-10 (3-class subset)",
                f"classes: {cfg.classes}",
                f"seed: {cfg.seed}",
                f"train_per_class: {cfg.n_train_per_class}",
                f"val_per_class: {cfg.n_val_per_class}",
                f"test_per_class: {cfg.n_test_per_class}",
                f"raw_dir: {cfg.raw_dir}",
                f"out_dir: {cfg.out_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Done.")
    print(f"Wrote images to: {cfg.out_dir.resolve()}")
    print(f"Metadata: {meta.resolve()}")


if __name__ == "__main__":
    main()

