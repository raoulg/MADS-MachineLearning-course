"""Your study of lesson 3, over the design in `designs.py`: a baseline, a search over the
capacity axes in which every draw of the route stage is paired with its capacity-matched
control, the winner rerun, the paired curves, the use of the experts in the route arm,
and the winner's route stage against its control, in points.

Run it from the root of the course repository, once Part 1 of `05_exercise.ipynb` has
run on this machine (it measures what one trial costs here):

    make study LESSON=3

The runs go to the Optuna studies `lesson3-study-<STUDY>` (the search) and
`lesson3-study-<STUDY>-runs` (the baseline, the winner, the route arm and its control),
so `make share LESSON=3` hands them in. Every run carries `designs.py` as its design
script, its `n_params` and its whole confusion matrix, and every run with a router the
use of its experts (`expert_share`, `expert_weight`, `expert_peak`). The budget is what
a trial of `SPACE` costs, read from Part 1's search per epoch and per training example.
The search maximises `OBJECTIVE`,
and every other metric can be read from the runs later. The study itself is
`hpo.exercise.Exercise`, the same in every lesson; `harness.py` is what lesson 3 gives
it, with `RoutedExercise` pairing the trials and writing the use of the experts on
every trial.
"""

import sys
from pathlib import Path

import datasets
import designs  # ty: ignore[import-not-found]  (a script's own folder is on sys.path)
import optuna
from designs import DESIGN  # ty: ignore[import-not-found]
from harness import (  # ty: ignore[import-not-found]
    EPOCHS,
    HARNESS,
    RoutedExercise,
    read,
)
from huggingface_hub.utils import disable_progress_bars as hub_quiet
from loguru import logger
from torch import nn

from hpo.exercise import Data, Questions
from hpo.record import Idea
from hpo.search import Choice, Config, Float, Int, Space

# What is your study called? Your first name, a dash, and what it studies, in lowercase
# letters, digits and dashes: "sam-experts-on-little-data".
STUDY = "name-your-study"

# What does your study investigate, in a few words of your own? Which idea does it build
# on? Your first one builds on the exercise as handed out; after that, name an idea of
# the class as the class page shows it: builds_on="blissful-bee-18".
IDEA = Idea.of("say what your study investigates", builds_on=Idea.handed_out(3).name)

# What do you expect to happen, and why? One line, written before you run: does `route`
# beat `matched`, by how many points, where on your curves, and will its router
# spread the positions over the experts or collapse onto one? It is stored with the study
# and never checked: it is there so the result can surprise you.
EXPECTS = "say what you expect, and why"

# Which metric does the search maximise? N is nine beats in ten, so "accuracy" rewards a
# model that says N; "macro_recall" counts a missed S as much as a missed N. Every run
# logs both, and every metric per class.
OBJECTIVE = "macro_recall"

# Which axes does your search space search, and over which ranges?
# EXAMPLE: the space below is E4's example space of 05_exercise.ipynb, set before any
# curve was measured. Replace its ranges with your own, from your curves:
# - Size, data, compute: where did a step of your curve over parameters (E2), over data
#   (E3) and over epochs (E4) pay, and by how many points beside the spread of its runs?
#   A stretch whose steps are the size of their spread is still open, not flat. An axis that still climbs at its limit (every beat,
#   the most epochs you can afford) is a finding about the budget.
# - The kind of study: one for the best design fixes an axis at its limit and searches
#   the axes that still move; one whose question is a curve ("does routing pay with less
#   data?") searches that axis below its limit too, and reads the paired difference
#   along it.
# - Fixed: which axes do you hold at one value, and why that value?
# - Control: "route": True with "matched": Choice([False, True]) pairs every draw: each
#   draw trains the route stage and its control of the same parameter count, on the same
#   width, depth, data, epochs and seed, and the readout is the difference within each
#   pair. Without `matched`, a gain of `route` says only that more parameters helped.
# Every option of designs.py (@option) needs an axis here: search it with
# Choice([False, True]), or fix it to one value. `balance` (designs.py) acts only with
# "routing": "sparse".
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
        "balance": 0.0,
        "parallel": False,
        "fold": False,
        "skip": False,
        "excite": False,
        "state": False,
        "route": True,
        "matched": Choice([False, True]),
        "gate": False,
        "class_weight": 0.0,
        "train_fraction": Float(0.05, 1.0, log=True),
        "lr": 1e-3,
        "epochs": Int(max(1, EPOCHS // 4), EPOCHS),
    }
)

# How many minutes do you give this study, on this machine? The trials come in pairs, so
# the pairs are half the trials; the budget cell of 05_exercise.ipynb shows how many
# trials your minutes buy.
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
    exercise = RoutedExercise(
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
