from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

from src.model import ModelConfig, SmallCNN


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path = Path("data/processed/cifar3")
    batch_size: int = 64
    epochs: int = 5
    num_workers: int = 2
    seed: int = 42

    # hyperparameters die we willen tunen
    lr: float = 1e-3
    weight_decay: float = 0.0

    # model knobs (ook tunable, maar voor nu default)
    base_channels: int = 16
    fc_units: int = 64
    dropout: float = 0.0

    # logging
    log_root: Path = Path("runs/grid")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_training(cfg: TrainConfig, run_name: str | None = None) -> dict:
    """
    Voert precies één trainingsrun uit en geeft de eind-metrics terug.
    Dit is de functie die hypertune.py steeds opnieuw aanroept.
    """
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = Compose([ToTensor()])
    train_ds = ImageFolder(cfg.data_dir / "train", transform=tfm)
    val_ds = ImageFolder(cfg.data_dir / "val", transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    mcfg = ModelConfig(
        num_classes=len(train_ds.classes),
        base_channels=cfg.base_channels,
        fc_units=cfg.fc_units,
        dropout=cfg.dropout,
    )
    model = SmallCNN(mcfg).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # TensorBoard
    if run_name is None:
        run_name = f"run_{int(time.time())}"
    log_dir = cfg.log_root / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        best_val_acc = max(best_val_acc, val_acc)
        best_val_loss = min(best_val_loss, val_loss)

    writer.close()

    result = {
        "run_name": run_name,
        "device": str(device),
        "classes": ",".join(train_ds.classes),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        **asdict(cfg),
    }
    return result


def main() -> None:
    # Eén run om te checken dat alles nog werkt na het splitsen
    cfg = TrainConfig(epochs=2, lr=1e-3, weight_decay=0.0, fc_units=64)
    res = run_training(cfg, run_name="single_run_check")
    print("Done. Result:")
    for k in ("run_name", "best_val_acc", "best_val_loss", "lr", "weight_decay", "fc_units"):
        print(f"{k}: {res[k]}")
    print("\nTensorBoard:")
    print(f"  uv run tensorboard --logdir {cfg.log_root}")


if __name__ == "__main__":
    main()
