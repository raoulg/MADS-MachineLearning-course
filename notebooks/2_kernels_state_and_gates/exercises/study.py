"""Your study of lesson 2, over the design in `designs.py`: a baseline, a search, the
winner rerun, and the difference in sigmas.

Run it from the root of the course repository, once Part 1 of `04_exercise.ipynb` has
run on this machine (it measures what one trial costs here):

    uv run notebooks/2_kernels_state_and_gates/exercises/study.py

The runs go to the Optuna studies `lesson2-study-<STUDY>` (the search) and
`lesson2-study-<STUDY>-runs` (the baseline and the winner), so `make share LESSON=2`
hands them in. Every run carries `designs.py` as its design script and logs its whole
confusion matrix: the search maximises `OBJECTIVE`, and every other metric can be read
from the runs later. The study itself is `hpo.exercise.Exercise`, the same in every
lesson; `harness.py` is what lesson 2 gives it.
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
# letters, digits and dashes: "sam-state-after-kernels".
STUDY = "name-your-study"

# What does your study investigate, in a few words of your own? Which idea does it build
# on? Your first one builds on the exercise as handed out; after that, name an idea of
# the class as the class page shows it: builds_on="blissful-bee-18".
IDEA = Idea.of("say what your study investigates", builds_on=Idea.handed_out(2).name)

# What do you expect to happen, and why? One line, written before you run. It is stored
# with the study and never checked: it is there so the result can surprise you.
EXPECTS = "say what you expect, and why"

# Which metric does the search maximise? N is nine beats in ten, so "accuracy" rewards a
# model that says N; "macro_recall" counts a missed S as much as a missed N. Every run
# logs both, and every metric per class.
OBJECTIVE = "macro_recall"

# Which axes does your search space search, and over which ranges? Every option of
# designs.py (@option) needs an axis here: search it with Choice([False, True]), or fix it
# to one value. Which axes do the trials you can afford cover well enough to read?
SPACE = Space(
    {
        "dataset": "mitbih",
        "kernel_size": Choice([3, 5, 9]),
        "width": Int(16, 64, log=True),
        "depth": Int(1, 5),
        "downsample": Choice(["pool", "stride"]),
        "norm": "none",
        "sigma": 0.0,
        "dropout": Float(0.0, 0.5),
        "parallel": Choice([False, True]),
        "fold": False,
        "skip": Choice([False, True]),
        "excite": False,
        "state": False,
        "gate": Choice([False, True]),
        "class_weight": Float(0.0, 1.0),
        "lr": Float(3e-4, 3e-3, log=True),
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
    return DESIGN.build(config, n_classes=len(data.classes))


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
            75.0, n_startup_trials=5, n_warmup_steps=3
        ),
    )
    exercise.run()


if __name__ == "__main__":
    main()
