from mltrainer import rnn_models, metrics, Trainer, ReportTypes, TrainerSettings
from mltrainer.preprocessors import PaddedPreprocessor

from pathlib import Path
from ray import tune
import torch
import ray
from typing import Dict
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from loguru import logger

from filelock import FileLock

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float


def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    from mads_datasets import DatasetFactoryProvider, DatasetType

    data_dir = config["data_dir"]
    gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    preprocessor = PaddedPreprocessor()

    with FileLock(data_dir / ".lock"):
        # we lock the datadir to avoid parallel instances trying to
        # access the datadir
        streamers = gesturesdatasetfactory.create_datastreamer(
            batchsize=32, preprocessor=preprocessor
        )
        train = streamers["train"]
        valid = streamers["valid"]

    # we set up the metric
    # and create the model with the config
    accuracy = metrics.Accuracy()
    model = rnn_models.GRUmodel(config)

    trainersettings = TrainerSettings(
        epochs=50,
        metrics=[accuracy],
        logdir=Path("."),
        train_steps=len(train),  # type: ignore
        valid_steps=len(valid),  # type: ignore
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    # because we set reporttypes=[ReportTypes.RAY]
    # the trainloop wont try to report back to tensorboard,
    # but will report back with ray
    # this way, ray will know whats going on,
    # and can start/pause/stop a loop.
    # This is why we set earlystop_kwargs=None, because we
    # are handing over this control to ray.

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,  # type: ignore
        traindataloader=train.stream(),
        validdataloader=valid.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    trainer.loop()


if __name__ == "__main__":
    ray.init()

    data_dir = Path("data/raw/gestures/gestures-dataset").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()

    config = {
        "input_size": 3,
        "output_size": 20,
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "hidden_size": tune.randint(16, 128),
        "dropout": tune.uniform(0.0, 0.3),
        "num_layers": tune.randint(2, 5),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        local_dir=str(config["tune_dir"]),
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()
