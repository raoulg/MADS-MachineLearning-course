from src.data import make_dataset
from src.models import rnn_models, metrics
from src.settings import SearchSpace
from pathlib import Path
from ray.tune import JupyterNotebookReporter
from ray import tune
import torch
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from loguru import logger
from filelock import FileLock

def train(config: SearchSpace, checkpoint_dir=None):

    data_dir = config["data_dir"]
    with FileLock(data_dir / ".lock"):
        trainloader, testloader = make_dataset.get_gestures(data_dir=data_dir, split=0.8, batchsize=32)

    accuracy = metrics.Accuracy()
    model = rnn_models.GRUmodel(config)

    model = rnn_models.trainloop(
        model=model,
        epochs=50,
        learning_rate=1e-3,
        optimizer=torch.optim.Adam,
        loss_fn=torch.nn.CrossEntropyLoss(),
        metrics=[accuracy],
        train_dataloader=trainloader,
        test_dataloader=testloader,
        tunewriter=True,
    )
 

if __name__ == "__main__":
    ray.init()
    config = SearchSpace(
        input_size=3,
        output_size=20,
        tune_dir=Path("models/ray").absolute(),
        data_dir=Path("data/external/gestures-dataset").absolute()
    )


    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    # scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", 
    #                                 grace_period=1,
    #                                 reduction_factor=4,
    #                                 max_t=50)
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=4,
        stop_last_trials=False,
    )
    bohb_search = TuneBOHB()

    analysis = tune.run(train,
         config = config.dict(),
         metric="test_loss",
         mode="min",
         progress_reporter=reporter,
         local_dir=config.tune_dir,
         num_samples=50,
         search_alg=bohb_search,
         scheduler=bohb_hyperband,
         verbose=1)

    ray.shutdown()
