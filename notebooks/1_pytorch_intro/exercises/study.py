"""Your study of lesson 1, over the design in `designs.py`: a baseline, a search, the
winner rerun, and the difference in sigmas.

Run it from the root of the course repository, once Part 1 of `05_exercises.ipynb` has
run on this machine (it measures what one trial costs here):

    uv run notebooks/1_pytorch_intro/exercises/study.py

The runs go to the same Optuna studies as Part 2 of the notebook, `lesson1-study-<STUDY>`
for the search and `lesson1-study-<STUDY>-runs` for the baseline and the winner, so
`make share LESSON=1` hands them in. Every run carries `designs.py` as its design script.
"""

import sys
import time
from dataclasses import fields
from pathlib import Path

import datasets
import designs  # ty: ignore[import-not-found]  (a script's own folder is on sys.path)
import numpy as np
import optuna
import pandas as pd
import torch
from designs import Config, Head, options  # ty: ignore[import-not-found]
from harness import (  # ty: ignore[import-not-found]
    EPOCHS,
    LINEAR,
    REPEATS,
    check_space,
    compare,
    loaders,
    objective,
    run_params,
    run_spread,
    shape_of,
)
from huggingface_hub.utils import disable_progress_bars as hub_quiet
from loguru import logger
from notebooktester import param
from optunalog import (  # harness puts the lesson folder on sys.path
    TESTING,
    Choice,
    Float,
    Int,
    budget,
    check_study_name,
    handed_out,
    idea,
    log_run,
    search_study,
    space_line,
    storage,
    use_study,
)
from torch import nn

# What is your study called? Your first name, a dash, and what it studies, in lowercase
# letters, digits and dashes: "sam-gate-on-eurosat".
STUDY = check_study_name("name-your-study")

# What does your study investigate, in a few words of your own? Which idea does it build
# on? Your first one builds on the exercise as handed out; after that, name an idea of
# the class as the class page shows it: builds_on="blissful-bee-18".
IDEA = idea("say what your study investigates", builds_on=handed_out(1)["idea_name"])

# What do you expect to happen, and why? One line, written before you run. It is stored
# with the study and never checked: it is there so the result can surprise you.
EXPECTS = "say what you expect, and why"

# Which axes does your search space search, and over which ranges? Every option of
# designs.py (@option) needs an axis here: search it with Choice([False, True]), or fix it
# to one value. Which axes do the trials you can afford cover well enough to read?
SPACE = {
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

# How many minutes do you give this study, on this machine?
MINUTES = 15

DESIGN = Path(designs.__file__).resolve()
# Under notebooktester the placeholder names the notebook's Part 2 study as well, which
# holds another space; the tester's run of this script keeps a study of its own.
STUDY_NAME = f"lesson1-study-{STUDY}{'-script' if TESTING else ''}"
BASELINE_ENCODER = "dinov2-small"

logger.remove()
logger.add(sys.stderr, level="WARNING")
datasets.disable_progress_bars()
hub_quiet()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_model(config, dim: int, n_classes: int) -> nn.Module:
    return Head(Config.drawn(config), dim, n_classes)


def part1_baseline(dataset: str, encoder: str) -> tuple[np.ndarray, float] | None:
    """The baseline of Part 1 of the notebook, when it ran on `dataset` with `encoder`
    at these `EPOCHS`: its accuracy per repeat and its seconds in training steps."""
    if "lesson1-notebook-runs" not in optuna.get_all_study_names(storage()):
        return None
    runs = optuna.load_study(study_name="lesson1-notebook-runs", storage=storage())
    matches = [
        t
        for t in runs.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.user_attrs.get("role") == "baseline"
        and t.params.get("dataset") == dataset
        and t.params.get("encoder") == encoder
        and t.params.get("epochs") == EPOCHS
        and len(t.intermediate_values) > 1
    ]
    if not matches:
        return None
    last = max(matches, key=lambda t: t.number)
    scores = np.array(
        [last.intermediate_values[k] for k in sorted(last.intermediate_values)]
    )
    return scores, float(last.user_attrs.get("train_seconds", 0.0))


def baseline_on(dataset: str) -> tuple[np.ndarray, float]:
    """The linear head on `BASELINE_ENCODER`, taken from Part 1 when it matches, else
    measured here, `REPEATS` times."""
    found = part1_baseline(dataset, BASELINE_ENCODER)
    if found is not None:
        print(f"baseline : Part 1's, on {dataset} with {BASELINE_ENCODER}")
        return found
    print(f"baseline : measured on {dataset} with {BASELINE_ENCODER}")
    dim, n_classes = shape_of(dataset, BASELINE_ENCODER)
    tr, te = loaders(dataset, BASELINE_ENCODER)
    return run_spread(lambda: nn.Linear(dim, n_classes), tr, te)


def main() -> None:
    check_space(SPACE, options(Head), [f.name for f in fields(Config)])

    try:
        spend = budget(
            MINUTES,
            like="lesson1-notebook",
            epochs=EPOCHS,
            repeats=REPEATS,
            space=SPACE,
        )
    except ValueError as error:
        raise ValueError(
            "budget: run Part 1 of notebooks/1_pytorch_intro/05_exercises.ipynb on this "
            f"machine first; it measures what one trial costs here ({error})"
        ) from None
    n_trials = param(spend["n_trials"], test=2)
    print(
        f"budget   : {spend['n_trials']} trials in {MINUTES} minutes at "
        f"{spend['seconds_per_trial']}s each, {spend['trials_per_axis']} per searched axis"
    )
    if n_trials == 0:
        raise ValueError(
            f"budget: {MINUTES} minutes hold no trial on this machine once the baseline "
            "and the winner are set aside; give the study more minutes or fewer epochs"
        )

    # the baseline of every dataset the space can draw, before the search
    drawn = SPACE["dataset"]
    on = drawn.choices if isinstance(drawn, Choice) else [drawn]
    baselines = {str(name): baseline_on(str(name)) for name in on}

    study = search_study(
        STUDY_NAME,
        SPACE,
        pruner=optuna.pruners.PercentilePruner(
            75.0, n_startup_trials=5, n_warmup_steps=4
        ),
        attrs={
            "origin": "study",
            "study": STUDY,
            "budget": {**spend, "n_trials": n_trials},
            # a study over a design of your own never searches the handed-out space
            "space_is_default": False,
            "expects": EXPECTS,
        },
        idea=IDEA,
    )
    start = time.time()
    study.optimize(
        objective(SPACE, make_model, DESIGN), n_trials=n_trials, show_progress_bar=True
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    table = pd.DataFrame([{"Accuracy": t.value, **t.params} for t in done])
    print(
        f"{n_trials} trials in {time.time() - start:.0f}s; {len(done)} finished in all"
    )
    print(table.sort_values("Accuracy", ascending=False).head(6).to_string(index=False))

    best = study.best_trial.user_attrs["config"]
    dataset, encoder = best["dataset"], best["encoder"]
    runs = use_study(f"{STUDY_NAME}-runs", idea=IDEA)

    baseline, baseline_seconds = baselines[dataset]
    base_dim, base_classes = shape_of(dataset, BASELINE_ENCODER)
    log_run(
        "baseline",
        "baseline",
        dataset=dataset,
        scores={"accuracy": baseline.tolist()},
        params=run_params(dataset, BASELINE_ENCODER, LINEAR),
        train_seconds=baseline_seconds,
        study=runs,
        model=nn.Linear(base_dim, base_classes),
        example=torch.zeros(2, base_dim),
        design=DESIGN,
    )

    dim, n_classes = shape_of(dataset, encoder)
    tr, te = loaders(dataset, encoder, n_train=best.get("n_train"))
    winner, winner_seconds = run_spread(
        lambda: make_model(best, dim, n_classes),
        tr,
        te,
        lr=best["lr"],
        epochs=best["epochs"],
    )
    log_run(
        "winner",
        "final",
        dataset=dataset,
        scores={"accuracy": winner.tolist()},
        params=run_params(dataset, encoder, best),
        train_seconds=winner_seconds,
        study=runs,
        model=make_model(best, dim, n_classes),
        example=torch.zeros(2, dim),
        attrs={"search_space": space_line(SPACE)},
        design=DESIGN,
    )

    print()
    print("winner   :", {k: v for k, v in best.items() if k != "epochs"})
    compare(f"baseline on {dataset}", baseline)
    sigmas = compare("your search winner", winner, versus=baseline)
    print(f"\nthe winner against the baseline: {sigmas:+.1f} sigma")
    print(f"idea     : {IDEA['idea_name']}  ({IDEA['idea']})")
    print(f"studies  : {STUDY_NAME}, {STUDY_NAME}-runs  (make optuna-dashboard)")


if __name__ == "__main__":
    main()
