"""Your study of lesson 3, over the design in `designs.py`: a baseline, a search over the
capacity axes and the route stage, the winner rerun, the capacity curves per arm, and the
winner's route stage against its capacity-matched control, in sigmas.

Run it from the root of the course repository, once Part 1 of `05_exercise.ipynb` has
run on this machine (it measures what one trial costs here):

    uv run notebooks/3_scaling_and_search/exercises/study.py

The runs go to the Optuna studies `lesson3-study-<STUDY>` (the search) and
`lesson3-study-<STUDY>-runs` (the baseline, the winner, the route arm and its control),
so `make share LESSON=3` hands them in. Every run carries `designs.py` as its design
script, its `n_params` and its whole confusion matrix: the search maximises `OBJECTIVE`,
and every other metric can be read from the runs later. The study itself is
`hpo.exercise.Exercise`, the same in every lesson; `harness.py` is what lesson 3 gives
it.
"""

import sys
from pathlib import Path

import datasets
import designs  # ty: ignore[import-not-found]  (a script's own folder is on sys.path)
import optuna
from designs import DESIGN  # ty: ignore[import-not-found]
from harness import EPOCHS, HARNESS, read  # ty: ignore[import-not-found]
from huggingface_hub.utils import disable_progress_bars as hub_quiet
from loguru import logger
from torch import nn

from hpo.exercise import Data, Exercise, Questions
from hpo.record import Idea
from hpo.search import Choice, Config, Float, Int, Space

# What is your study called? Your first name, a dash, and what it studies, in lowercase
# letters, digits and dashes: "sam-experts-on-little-data".
STUDY = "name-your-study"

# What does your study investigate, in a few words of your own? Which idea does it build
# on? Your first one builds on the exercise as handed out; after that, name an idea of
# the class as the class page shows it: builds_on="blissful-bee-18".
IDEA = Idea.of("say what your study investigates", builds_on=Idea.handed_out(3).name)

# What do you expect to happen, and why? One line, written before you run. It is stored
# with the study and never checked: it is there so the result can surprise you.
EXPECTS = "say what you expect, and why"

# Which metric does the search maximise? N is nine beats in ten, so "accuracy" rewards a
# model that says N; "macro_recall" counts a missed S as much as a missed N. Every run
# logs both, and every metric per class.
OBJECTIVE = "macro_recall"

# Which axes does your search space search, and over which ranges? A capacity curve needs
# at least two of the three capacity axes: parameters (width, depth), data
# (train_fraction) and compute (epochs). Every option of designs.py (@option) needs an
# axis here: search it with Choice([False, True]), or fix it to one value. `matched`
# replaces the route stage by its control of the same parameter count; searched next to
# `route`, the trials hold both arms, so the curves can be read per arm. Which axes do
# the trials you can afford cover well enough to read?
SPACE = Space(
    {
        "dataset": "mitbih",
        "kernel_size": 5,
        "width": Int(8, 64, log=True),
        "depth": Int(1, 4),
        "downsample": "pool",
        "norm": "none",
        "sigma": 0.0,
        "dropout": 0.0,
        "n_experts": 4,
        "routing": "dense",
        "parallel": False,
        "fold": False,
        "skip": False,
        "excite": False,
        "state": False,
        "route": Choice([False, True]),
        "matched": Choice([False, True]),
        "gate": False,
        "class_weight": 0.0,
        "train_fraction": Float(0.05, 1.0, log=True),
        "lr": 1e-3,
        "epochs": Int(max(1, EPOCHS // 4), EPOCHS),
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
    # No pruner: a pruner stops a trial on its first epochs, and a capacity curve
    # compares runs that each ran their whole budget.
    exercise = Exercise(
        HARNESS,
        questions,
        SPACE,
        DESIGN,
        make_model,
        Path(designs.__file__).resolve(),
    )
    outcome = exercise.run()
    read(exercise, outcome)


if __name__ == "__main__":
    main()
