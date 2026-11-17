"""Made-up submissions of a class, for looking at `class_analysis.py` before any student
has submitted: eight invented students, each with the handed-out baseline, the handed-out
search of 24 trials and its winner, and most of them with a study of their own.

Every study carries an idea and says what its author expects: the course's handed-out
idea is the root, the students' own ideas build on it, one idea builds on two
classmates' ideas (a merge) and one on that, and one names a parent that nobody
submitted. Every trial logs its curves per epoch as a trial of `hpo` does, a pruned one
for the epochs it ran; every study carries its budget and its design script, whose
`@option("skip")` the winner has on or off.

The accuracy of a trial is a plain function of its params plus noise between runs, so the
pictures show what real runs would: a trend along the learning rate, a depth that stops
paying, a spread between runs that is larger than most gains.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

import numpy as np
from run_submission import COURSE, IDEA_ATTRS, Submission, idea_id, idea_name

NAMES = ["alma", "bram", "chen", "dina", "emre", "fenna", "gijs", "hana"]
NOISE = 0.0035  # the spread between runs of one config, as a standard deviation
ACT = {"relu": 0.0, "gelu": 0.002, "tanh": -0.004, "sigmoid": -0.012}
EPOCHS = 15
PRUNED_AFTER = 5  # the epochs a pruned trial ran


def accuracy(cfg: dict, rng: np.random.Generator, machine: float) -> float:
    """The expected accuracy of a head config on DTD, with the noise of one run."""
    acc = 0.79 + machine
    if cfg["n_hidden_layers"] > 0:
        acc += 0.012 - 0.006 * (cfg["n_hidden_layers"] - 1)
        acc += 0.004 * math.log10(cfg["hidden_dim"] / 64)
        acc += ACT[cfg["activation"]]
        acc += 0.003 * (cfg["norm"] != "none") + 0.002 * cfg["skip"]
    acc -= 0.03 * abs(math.log10(cfg["lr"] / 5e-4)) ** 1.5
    if cfg["n_hidden_layers"] > 0:
        acc -= 0.04 * abs(cfg["dropout"] - 0.25) ** 1.2
    acc -= 0.03 * cfg["sigma"]
    return float(acc + rng.normal(0, NOISE))


def curve(
    final: float, lr: float, rng: np.random.Generator, epochs: int
) -> list[float]:
    """The accuracy after each of `epochs` epochs of a trial that would end at `final`
    after `EPOCHS`: a rise that is quicker for a larger learning rate, with noise."""
    rate = 1.2 + 1.5 * max(0.0, math.log10(lr / 1e-4))
    values = [
        final * (1 - 0.32 * math.exp(-s / rate)) + float(rng.normal(0, 0.0015))
        for s in range(EPOCHS)
    ]
    values[-1] = final
    return [round(v, 5) for v in values[:epochs]]


def make_idea(author: str, words: str, parents: list[dict[str, str]]) -> dict[str, str]:
    """The study attrs of an idea, built as `hpo.record.ideas.Idea.of` builds them."""
    names = sorted(p["idea_name"] for p in parents)
    id_ = idea_id(author, words, names)
    values = (words, author, ",".join(names), id_, idea_name(id_))
    return dict(zip(IDEA_ATTRS, values, strict=True))


ROOT = make_idea(COURSE, "a head on frozen dinov2 features for dtd", [])


def params_of(cfg: dict) -> dict:
    """What a run logs: a head without hidden layers logs the hidden block as off."""
    p = {"encoder": "dinov2-small", "embed_dim": 384, "epochs": EPOCHS, **cfg}
    if cfg["n_hidden_layers"] == 0:
        p.update(hidden_dim=0, activation="none", norm="none", dropout=0.0, skip=False)
    return p


def n_params(cfg: dict) -> int:
    if cfg["n_hidden_layers"] == 0:
        return 384 * 47 + 47
    h, n = cfg["hidden_dim"], cfg["n_hidden_layers"]
    return 384 * h + h + (n - 1) * (h * h + h) + h * 47 + 47


def distribution(low, high, log=False, integer=False) -> dict:
    return {
        "name": "IntDistribution" if integer else "FloatDistribution",
        "attributes": {
            "low": low,
            "high": high,
            "log": log,
            "step": 1 if integer else None,
        },
    }


SPACE = {
    "n_hidden_layers": distribution(0, 3, integer=True),
    "hidden_dim": distribution(32, 1024, log=True, integer=True),
    "activation": {
        "name": "CategoricalDistribution",
        "attributes": {"choices": ["relu", "gelu", "tanh", "sigmoid"]},
    },
    "norm": {
        "name": "CategoricalDistribution",
        "attributes": {"choices": ["none", "batch", "layer"]},
    },
    "skip": {
        "name": "CategoricalDistribution",
        "attributes": {"choices": [False, True]},
    },
    "dropout": distribution(0.0, 0.5),
    "sigma": distribution(0.0, 0.5),
    "lr": distribution(1e-4, 1e-2, log=True),
}
SPACE_LINE = (
    "dataset=dtd | encoder=dinov2-small | n_hidden_layers ~ int(0, 3) | "
    "hidden_dim ~ logint(32, 1024) | activation in {relu,gelu,tanh,sigmoid} | "
    "norm in {none,batch,layer} | skip in {false,true} | dropout ~ U(0, 0.5) | "
    "sigma ~ U(0, 0.5) | lr ~ logU(1e-4, 1e-2)"
)
# The values a study's space fixes when it does not search an axis.
FIXED = {"n_hidden_layers": 1, "hidden_dim": 256, "activation": "relu", "norm": "none",
         "skip": False, "dropout": 0.25, "sigma": 0.0, "lr": 5e-4}  # fmt: skip


def space_line(searched: list[str]) -> str:
    """A study's space in one line, as `hpo.search.space.Space.line` writes it."""
    full: dict[str, str] = {
        name: rest
        for name, _, rest in (
            part.partition(" ") for part in SPACE_LINE.split(" | ")[2:]
        )
    }
    axes = [
        f"{a} {full[a]}" if a in searched else f"{a}={str(FIXED[a]).lower()}"
        for a in FIXED
    ]
    return " | ".join(["dataset=dtd", "encoder=dinov2-small", *axes])


def draw(rng: np.random.Generator) -> dict:
    return {
        "n_hidden_layers": int(rng.integers(0, 4)),
        "hidden_dim": round(math.exp(rng.uniform(math.log(32), math.log(1024)))),
        "activation": str(rng.choice(["relu", "gelu", "tanh", "sigmoid"])),
        "norm": str(rng.choice(["none", "batch", "layer"])),
        "skip": bool(rng.integers(0, 2)),
        "dropout": float(rng.uniform(0, 0.5)),
        "sigma": float(rng.uniform(0, 0.5)),
        "lr": float(math.exp(rng.uniform(math.log(1e-4), math.log(1e-2)))),
    }


# The classes of DTD the demo logs per class: a few, so the page can be read on one.
DEMO_CLASSES = ("banded", "blotchy", "braided")


def logged(value: float | None, std: float | None) -> dict:
    """Every metric a run logs, made up from its accuracy: the macro recall a little
    below it, and the recall of a few classes around it."""
    if value is None:
        return {}
    recall = {c: round(min(1.0, max(0.0, value + d)), 5)
              for c, d in zip(DEMO_CLASSES, (0.08, -0.05, -0.15), strict=True)}  # fmt: skip
    metrics = {
        "accuracy": value,
        "macro_recall": round(max(0.0, value - 0.03), 5),
        "recall": recall,
        "precision": recall,
        "f1": recall,
    }
    out: dict = {"metrics": metrics}
    if std is not None:
        spread = {c: std for c in DEMO_CLASSES}
        out["metrics_std"] = {"accuracy": std, "macro_recall": std, "recall": spread,
                              "precision": spread, "f1": spread}  # fmt: skip
    return out


def curves(values: list[float]) -> dict:
    """The curves a trial logs as user attrs: accuracy and macro recall per epoch."""
    return {
        "accuracy": values,
        "macro_recall": [round(max(0.0, v - 0.03), 5) for v in values],
    }


def design(student: str, searched: list[str]) -> str:
    """A design script of the shape a lesson's `designs.py` has, with one optional
    stage."""
    return (
        f'"""The head of {student}\'s study: one method per stage, the skip optional."""\n\n'
        "from torch import Tensor, nn\n"
        "from vectormesh.design import option\n\n\n"
        "class Head(nn.Module):\n"
        "    def hidden(self, x: Tensor) -> Tensor:  # (batch, embed) -> (batch, hidden)\n"
        "        return self.block(x)\n\n"
        '    @option("skip")\n'
        "    def skip(self, x: Tensor, h: Tensor) -> Tensor:  # adds the input back\n"
        "        return h + self.project(x)\n\n"
        "    def classify(self, h: Tensor) -> Tensor:  # (batch, hidden) -> (batch, classes)\n"
        "        return self.out(h)\n"
        f"# searched: {', '.join(searched)}\n"
    )


# Per student's study: its name, the axes it searches, what it adds to the accuracy,
# the words of its idea and what its author expects.
STUDIES = {
    "alma": ("alma-wide-head", ["hidden_dim", "n_hidden_layers", "lr"], 0.012,
             "does a wider head still help on dtd",
             "wider helps up to about 512, then it flattens"),
    "bram": ("bram-norm-or-skip", ["norm", "skip", "n_hidden_layers", "lr"], 0.006,
             "norm or skip, which one carries the gain",
             "the norm, since it steadies the larger heads"),
    "chen": ("chen-small-lr-range", ["lr", "dropout"], -0.004,
             "a narrow learning rate range around the default",
             "the best lr sits near 5e-4, a narrow range finds it faster"),
    "dina": ("dina-wide-heads", ["hidden_dim", "lr"], 0.018,
             "very wide heads with one hidden layer",
             "a very wide head adds a point over 256"),
    "emre": ("emre-sigma-noise", ["sigma", "lr"], 0.0,
             "input noise as the only regulariser",
             "noise alone regularises as well as dropout"),
    "fenna": ("fenna-pruner-first", list(FIXED), 0.009,
              "prune early and search the whole space",
              "pruning lets the whole space fit the budget"),
}  # fmt: skip


def demo_submissions(lesson: int = 1, seed: int = 7) -> list[Submission]:
    rng = np.random.default_rng(seed)
    subs: list[Submission] = []
    clock = [datetime(2026, 11, 20, 18, tzinfo=UTC).timestamp() * 1000]
    finals: dict[str, str] = {}

    def add(student, experiment, name, role, value, *, params, attrs, study_attrs,
            distributions=None, status="COMPLETE", intermediate=None, score=None,
            artifacts=None):  # fmt: skip
        clock[0] += 60_000
        run_id = uuid.UUID(
            int=int(rng.integers(0, 2**62)) << 64 | int(rng.integers(0, 2**62))
        ).hex
        sub = Submission.model_validate(
            {
                "schema": "mads-run/2",
                "lesson": lesson,
                "student": student,
                "submitted_at": datetime(2026, 11, 22, tzinfo=UTC),
                "code": {"commit": None, "dirty": None, "remote": None},
                "run": {
                    "run_id": run_id,
                    "run_name": name,
                    "experiment": experiment,
                    "status": status,
                    "start_time": int(clock[0]),
                    "end_time": int(clock[0]) + 5_000,
                    "params": params,
                    "distributions": distributions or {},
                    "value": value,
                    "intermediate": intermediate or [],
                    "attrs": {
                        "role": role,
                        "dataset": "dtd",
                        "session": "2026-11-20 18:00:00",
                        **logged(
                            value if score is None else score, attrs.get("accuracy_std")
                        ),
                        **attrs,
                    },
                    "study": {"direction": "maximize", "attrs": study_attrs},
                    "artifacts": artifacts or {},
                },
            }
        )
        subs.append(sub)
        return sub.ref()

    def repeats(cfg: dict, machine: float, n: int = 3) -> tuple[float, float]:
        scores = np.array([accuracy(cfg, rng, machine) for _ in range(n)])
        return float(scores.mean()), float(scores.std())

    def trial(student, experiment, i, cfg, acc, searched, sattrs, pruned=False):
        """One trial of a search: its curve to the end, or to where the pruner stopped
        it, its params on the axes the study searches and its config."""
        values = curve(acc, cfg["lr"], rng, PRUNED_AFTER if pruned else EPOCHS)
        add(student, experiment, f"trial-{i:02d}", "trial", None if pruned else values[-1],
            score=values[-1], params={k: cfg[k] for k in searched},
            distributions={k: SPACE[k] for k in searched},
            status="PRUNED" if pruned else "COMPLETE",
            intermediate=[{"step": s, "value": v} for s, v in enumerate(values)],
            attrs={"n_params": n_params(cfg), "config": {"dataset": "dtd", **cfg},
                   **curves(values)},
            study_attrs=sattrs)  # fmt: skip

    linear = {"n_hidden_layers": 0, "hidden_dim": 0, "activation": "relu", "norm": "none",
              "skip": False, "dropout": 0.0, "sigma": 0.0, "lr": 1e-3}  # fmt: skip

    def study(student, name, searched, edge, idea, expects, machine, builds_on=None):
        """A student's study: the baseline, a search over `searched` that pays `edge`,
        and its winner rerun, as `hpo.exercise.study.Exercise` logs them."""
        experiment = f"lesson{lesson}-study-{name}"
        n_trials = int(rng.integers(12, 25))
        sattrs = {
            "search_space": space_line(searched),
            "searched_axes": ",".join(sorted(searched)),
            "origin": "study",
            "study": name,
            "space_is_default": False,
            "expects": expects,
            "budget": {"minutes": round(n_trials * 0.09 + 0.4, 1),
                       "seconds_per_trial": 5.0, "measured_on": f"lesson{lesson}-notebook",
                       "machine": "demo", "n_trials": n_trials, "repeats": 3,
                       "epochs": EPOCHS},
            **idea,
        }  # fmt: skip
        rattrs = {"origin": "study", "study": name, **idea}
        script = design(student, searched)
        base_mean, base_std = repeats(linear, machine)
        add(student, f"{experiment}-runs", "baseline", "baseline", base_mean, params=params_of(linear),
            attrs={"accuracy_std": base_std, "n_repeats": 3, "n_params": n_params(linear), "name": "baseline"},
            study_attrs=rattrs, artifacts={"design.py": script})  # fmt: skip
        best_cfg, best_acc = linear, base_mean
        for i in range(n_trials):
            drawn = draw(rng)
            cfg = {**FIXED, **{a: drawn[a] for a in searched}}
            acc = accuracy(cfg, rng, machine) + edge * 0.3
            pruned = acc < base_mean - 0.01 and i >= 5
            trial(student, experiment, i, cfg, acc, searched, sattrs, pruned)
            if not pruned and acc > best_acc:
                best_cfg, best_acc = cfg, acc
        f_mean, f_std = repeats(best_cfg, machine)
        f_mean += edge * 0.3
        extra = {"builds_on": builds_on} if builds_on else {}
        add(student, f"{experiment}-runs", "winner", "final", f_mean, params=params_of(best_cfg),
            attrs={"accuracy_std": f_std, "n_repeats": 3, "n_params": n_params(best_cfg),
                   "name": "winner", "search_space": sattrs["search_space"], **extra},
            study_attrs=rattrs, artifacts={"design.py": script})  # fmt: skip
        return subs[-1].ref()

    own_ideas = {s: make_idea(s, v[3], [ROOT]) for s, v in STUDIES.items()}
    machines = {}
    for student in NAMES:
        machine = machines[student] = float(rng.normal(0, 0.004))
        mean, std = repeats(linear, machine)
        # handed-out part
        add(student, f"lesson{lesson}-notebook-runs", "baseline", "baseline", mean,
            params=params_of(linear), attrs={"accuracy_std": std, "n_repeats": 3, "n_params": n_params(linear), "name": "baseline"},
            study_attrs=ROOT)  # fmt: skip
        notebook = {"search_space": SPACE_LINE, "searched_axes": ",".join(sorted(SPACE)),
                    "origin": "notebook", **ROOT}  # fmt: skip
        for i in range(24):
            cfg = draw(rng)
            trial(student, f"lesson{lesson}-notebook", i, cfg, accuracy(cfg, rng, machine),
                  list(SPACE), notebook)  # fmt: skip
        if student in STUDIES:
            name, searched, edge, _, expects = STUDIES[student]
            finals[student] = study(
                student, name, searched, edge, own_ideas[student], expects, machine
            )
    # Ideas that build on classmates' ideas: dina merges alma's and bram's, hana builds
    # on dina's merge, bram on an idea of gijs that gijs never submitted. Their winners
    # name the run they compared with in builds_on.
    unsubmitted = make_idea("gijs", "a deeper head with layer norm", [ROOT])
    merged = make_idea("dina", "a wide head with the norm that carried bram's gain",
                       [own_ideas["alma"], own_ideas["bram"]])  # fmt: skip
    later = [
        ("bram", "bram-deeper-norm", ["n_hidden_layers", "norm", "lr"], 0.008,
         make_idea("bram", "layer norm on a deeper head", [unsubmitted]),
         "layer norm makes the second layer pay", None),
        ("dina", "dina-wide-and-norm", ["hidden_dim", "norm", "skip", "lr"], 0.024, merged,
         "the two gains add up", "alma"),
        ("hana", "hana-merged-less-data", ["hidden_dim", "lr"], 0.016,
         make_idea("hana", "the merged wide head on less data", [merged]),
         "half the data costs less than a point", "dina"),
    ]  # fmt: skip
    for student, name, searched, edge, idea, expects, on in later:
        finals[student] = study(
            student, name, searched, edge, idea, expects, machines[student],
            finals[on] if on else None,
        )  # fmt: skip
    return subs
