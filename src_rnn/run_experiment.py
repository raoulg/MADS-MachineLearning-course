# src_rnn/run_experiment.py
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Dict, Any

import mlflow
import torch
from torch import optim

from mltrainer import Trainer, TrainerSettings

from .models import RNNConfig, build_model


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
    tracking_uri: str = "sqlite:///mlflow.db",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    extra_tags: Optional[Dict[str, str]] = None,
) -> torch.nn.Module:
    """
    Start één reproduceerbare training-run met MLflow logging.

    Notebooks blijven dun: zij geven alleen config + streamers + settings door.
    """
    # MLflow setup
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Build model
    model = build_model(model_name=model_name, config=config).to(device)

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Tags (handig voor filteren)
        mlflow.set_tag("model_name", model_name)
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

        # Params loggen
        for k, v in asdict(config).items():
            mlflow.log_param(k, v)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("device", str(device))
        mlflow.log_param("epochs", getattr(settings, "epochs", None))

        # Trainer (mltrainer verwacht optimizer als class; kwargs via optimizer_kwargs)
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

        # Probeer optimizer_kwargs te gebruiken (meest correct). Als mltrainer dit niet ondersteunt,
        # valt het terug op Adam defaults zodat je pipeline blijft werken.
        try:
            trainer_kwargs["optimizer_kwargs"] = {"lr": lr, "weight_decay": weight_decay}
            trainer = Trainer(**trainer_kwargs)
        except TypeError:
            # Fallback: Trainer accepteert geen optimizer_kwargs
            trainer_kwargs.pop("optimizer_kwargs", None)
            trainer = Trainer(**trainer_kwargs)

        trainer.loop()

    return model

