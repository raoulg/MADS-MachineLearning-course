# src_rnn/run_experiment.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional, Dict

import mlflow
import torch
from torch import optim

from mltrainer import Trainer
from mltrainer import TrainerSettings

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

    # Loss + optimizer factory
    loss_fn = torch.nn.CrossEntropyLoss()

    def optimizer_factory(params):
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

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

        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer_factory,  # factory i.p.v. class, zodat lr/weight_decay vastligt
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )

        trainer.loop()

    return model
