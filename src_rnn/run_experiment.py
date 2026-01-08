# src_rnn/run_experiment.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import torch
from torch import optim

from mltrainer import Trainer, TrainerSettings

from .models import RNNConfig, build_model


def _default_tracking_uri() -> str:
    """
    Maak een stabiele, absolute sqlite tracking URI.
    Werkt ongeacht waar je notebook/terminal de cwd heeft.
    """
    project_root = Path(__file__).resolve().parents[1]  # .../MADS-MachineLearning-course
    db_path = project_root / "notebooks" / "ex3" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def run_experiment(
    *,
    model_name: str,
    config: RNNConfig,
    settings: TrainerSettings,
    trainstreamer,
    validstreamer,
    device,
    experiment_name: str = "gestures-ex3",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    extra_tags: Optional[Dict[str, str]] = None,
) -> torch.nn.Module:
    """
    Start één reproduceerbare training-run met MLflow logging.

    Notebooks blijven dun: zij geven alleen config + streamers + settings door.
    """
    # 1) MLflow tracking (stabiel pad)
    if tracking_uri is None:
        tracking_uri = _default_tracking_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 2) Build model
    model = build_model(model_name=model_name, config=config).to(device)

    # 3) Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # 4) Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("tracking_uri", tracking_uri)
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

        # Params
        for k, v in asdict(config).items():
            mlflow.log_param(k, v)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("device", str(device))
        mlflow.log_param("epochs", getattr(settings, "epochs", None))

        # Trainer
        trainer_kwargs: Dict[str, Any] = dict(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optim.Adam,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )

        # mltrainer: probeer optimizer_kwargs (als ondersteund)
        try:
            trainer_kwargs["optimizer_kwargs"] = {"lr": lr, "weight_decay": weight_decay}
            trainer = Trainer(**trainer_kwargs)
        except TypeError:
            # Fallback: Trainer accepteert geen optimizer_kwargs
            trainer_kwargs.pop("optimizer_kwargs", None)
            trainer = Trainer(**trainer_kwargs)

        trainer.loop()

    return model
