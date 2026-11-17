"""Your study of lesson 1, over the design in `designs.py`: a baseline, a search, the
winner rerun, and the difference in sigmas.

Run it from the root of the course repository, once Part 1 of `05_exercises.ipynb` has
run on this machine (it measures what one trial costs here):

    uv run notebooks/1_representation_and_head/exercises/study.py

The runs go to the Optuna studies `lesson1-study-<STUDY>` (the search) and
`lesson1-study-<STUDY>-runs` (the baseline and the winner), so `make share LESSON=1`
hands them in. Every run carries `designs.py` as its design script, and logs every
metric (`hpo.measure.metrics`): the search maximises `OBJECTIVE`, and the others are there for a
question you ask later. The study itself is `hpo.exercise.study.Exercise`, the same in every
lesson; `harness.py` is what lesson 1 gives it.
"""

import sys
from pathlib import Path

import datasets
import designs  # ty: ignore[import-not-found]  (a script's own folder is on sys.path)
import optuna
from designs import DESIGN  # ty: ignore[import-not-found]
from harness import EPOCHS, HARNESS  # ty: ignore[import-not-found]
from huggingface_hub.utils import disable_progress_bars as hub_quiet
from loguru import logger
from torch import nn

from hpo.exercise import Data, Exercise, Questions
from hpo.record import Idea
from hpo.search import Choice, Config, Float, Int, Space

# What is your study called? Your first name, a dash, and what it studies, in lowercase
# letters, digits and dashes: "sam-gate-on-eurosat".
STUDY = "name-your-study"

# What does your study investigate, in a few words of your own? Which idea does it build
# on? Your first one builds on the exercise as handed out; after that, name an idea of
# the class as the class page shows it: builds_on="blissful-bee-18".
IDEA = Idea.of("say what your study investigates", builds_on=Idea.handed_out(1).name)

# What do you expect to happen, and why? One line, written before you run. It is stored
# with the study and never checked: it is there so the result can surprise you.
EXPECTS = "say what you expect, and why"

# Which metric does the search maximise? "accuracy", or "macro_recall" when every class
# should count as much, however rare. Every run logs both, and every metric per class.
OBJECTIVE = "accuracy"

# Which axes does your search space search, and over which ranges? Every option of
# designs.py (@option) needs an axis here: search it with Choice([False, True]), or fix it
# to one value. Which axes do the trials you can afford cover well enough to read?
SPACE = Space(
    {
        "dataset": "dtd",
        "encoder": Choice(["dinov2-small"]),
        "n_hidden_layers": Int(1, 2),
        "hidden_dim": Int(32, 1024, log=True),
        "activation": Choice(["relu", "gelu"]),
        "norm": "none",
        "dropout": Float(0.0, 0.5),
        "sigma": 0.0,
        "skip": Choice([False, True]),
        "gate": Choice([False, True]),
        "lr": Float(1e-4, 1e-2, log=True),
        "epochs": EPOCHS,
    }
)

# How many minutes do you give this study, on this machine?
MINUTES = 15

logger.remove()
logger.add(sys.stderr, level="WARNING")
datasets.disable_progress_bars()
hub_quiet()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_model(config: Config, data: Data) -> nn.Module:
    return DESIGN.build(config, dim=data.dims["embed_dim"], n_classes=len(data.classes))


def main() -> None:
    questions = Questions(
        study=STUDY, idea=IDEA, expects=EXPECTS, objective=OBJECTIVE, minutes=MINUTES
    )
    exercise = Exercise(
        HARNESS,
        questions,
        SPACE,
        DESIGN,
        make_model,
        Path(designs.__file__).resolve(),
        pruner=optuna.pruners.PercentilePruner(
            75.0, n_startup_trials=5, n_warmup_steps=4
        ),
    )
    exercise.run()


if __name__ == "__main__":
    main()
