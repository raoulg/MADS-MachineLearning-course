from src.data import make_dataset
from src.models import rnn_models, metrics
from pathlib import Path
from ray.tune import JupyterNotebookReporter
from ray import tune
import torch
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from loguru import logger
# from filelock import FileLock

def train(config, checkpoint_dir=None):
    data_dir = config["data_dir"]
    if not data_dir.exist():
        logger.error(f"Datadir {data_dir} not found")
        raise FileNotFoundError

    accuracy = metrics.Accuracy()
    trainloader, testloader = make_dataset.get_gestures(data_dir=data_dir, split=0.8, batchsize=32)
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

    config = {
        "input_size" : 3,
        "hidden_size" : tune.randint(16, 128), 
        "dropout" : tune.uniform(0.0, 0.3),
        "num_layers": tune.randint(2, 4),
        "output_size" : 20,
        "tune_dir" : None,
        "data_dir" : Path("~/code/deep_learning/data/external/gestures-dataset").expanduser()
    }

    reporter = CLIReporter()
    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration", 
                                    grace_period=5,
                                    reduction_factor=3,
                                    max_t=50)

    local_dir = Path("~/code/deep_learning/models/").expanduser()

    analysis = tune.run(train,
         config = config,
         metric="test_loss",
         mode="min",
         progress_reporter=reporter,
         local_dir=local_dir,
         num_samples=10,
         stop={"training_iteration": 50},
         sync_config=tune.SyncConfig(syncer=None),
         verbose=1)

    ray.shutdown()
