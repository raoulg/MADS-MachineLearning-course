"""Your study of lesson 3, over the design in `designs.py`: a baseline, a search, the
winner rerun, and its difference with the baseline in points.

Run it from the root of the course repository, once Part 1 of `05_exercise.ipynb` has
run on this machine (it measures what one trial costs here):

    make study LESSON=3

The runs go to the Optuna studies `lesson3-study-<STUDY>` (the search) and
`lesson3-study-<STUDY>-runs` (the baseline and the winner), so `make share LESSON=3`
hands them in. Every run carries `designs.py` as its design script, its whole confusion
matrix, and what one image costs it: `flops_per_image` and `latency_us`, beside
`n_params`. The search maximises `OBJECTIVE`; accuracy against FLOPs, the Pareto front,
is read from the runs afterwards (`make optuna-dashboard`). The study itself is
`hpo.exercise.Exercise`, the same in every lesson; `harness.py` is what lesson 3 gives
it.
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
# letters, digits and dashes: "sam-kernel-on-the-fold".
STUDY = "name-your-study"

# What does your study investigate, in a few words of your own? `builds_on` names the
# idea yours starts from: the exercise as handed out (below), or an idea of the class by
# its name as the class page shows it: builds_on="blissful-bee-18".
IDEA = Idea.of("say what your study investigates", builds_on=Idea.handed_out(3).name)

# What do you expect to happen, and why? One line, written before you run. It is stored
# with the study and never checked: it is there so the result can surprise you.
EXPECTS = "say what you expect, and why"

# Which metric does the search maximise? "accuracy", or "macro_recall" when every class
# should count as much, however rare. Every run logs both, and every metric per class.
OBJECTIVE = "accuracy"

# Which axes does your search space search, and over which ranges? The space below is
# the starter space of Part 1 (E4): share, input, width, depth and dropout. Every option
# of designs.py (@option) needs an axis here: search it with Choice([False, True]), or
# fix it to one value. An option you search needs its control of equal compute among the
# trials: search `width` beside it over a range that holds the control's width (step 4
# of Part 2). The kernel reads the grid: with it, put "form": "grid". Which axes do the
# trials you can afford cover well enough to read?
SPACE = Space(
    {
        "dataset": "mix7",
        "encoder": "dinov2-small",
        "share": Float(0.1, 1.0, log=True),
        "drop_domain": "none",
        "form": "flat",
        "input": Choice(["cls", "mean", "both"]),
        "width": Int(16, 256, log=True),
        "depth": Int(1, 2),
        "norm": "none",
        "dropout": Float(0.0, 0.3),
        "kernel": False,
        "skip": False,
        "gate": False,
        "moe": False,
        "lr": 1e-3,
        "epochs": EPOCHS,
    }
)

# How many minutes do you give this study, on this machine?
MINUTES = 30

logger.remove()
logger.add(sys.stderr, level="WARNING")
datasets.disable_progress_bars()
hub_quiet()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_model(config: Config, data: Data) -> nn.Module:
    return DESIGN.build(
        config,
        dim=data.dims["dim"],
        height=data.dims["height"],
        breadth=data.dims["breadth"],
        n_classes=len(data.classes),
    )


def main() -> None:
    questions = Questions(
        study=STUDY, idea=IDEA, expects=EXPECTS, objective=OBJECTIVE, minutes=MINUTES
    )
    # No pruner: a small head learns slowest in its first epochs, and it is the small
    # heads the Pareto front asks about. Every run stops by itself once its validation
    # accuracy settles (harness.py).
    exercise = Exercise(
        HARNESS, questions, SPACE, DESIGN, make_model, Path(designs.__file__).resolve()
    )
    exercise.run()


if __name__ == "__main__":
    main()
