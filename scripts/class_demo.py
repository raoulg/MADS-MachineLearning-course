"""Made-up submissions of a class, for looking at `class_analysis.py` before any student
has submitted: eight invented students, each with the handed-out baseline, the handed-out
search of 24 trials and its winner, and most of them with a study of their own.

The accuracy of a trial is a plain function of its params plus noise between runs, so the
pictures show what real runs would: a trend along the learning rate, a depth that stops
paying, a spread between runs that is larger than most gains.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

import numpy as np
from run_submission import Submission

NAMES = ["alma", "bram", "chen", "dina", "emre", "fenna", "gijs", "hana"]
NOISE = 0.0035  # the spread between runs of one config, as a standard deviation
ACT = {"relu": 0.0, "gelu": 0.002, "tanh": -0.004, "sigmoid": -0.012}


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


def params_of(cfg: dict) -> dict:
    """What a run logs: a head without hidden layers logs the hidden block as off."""
    p = {"encoder": "dinov2-small", "embed_dim": 384, "epochs": 15, **cfg}
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


def demo_submissions(lesson: int = 1, seed: int = 7) -> list[Submission]:
    rng = np.random.default_rng(seed)
    subs: list[Submission] = []
    clock = [datetime(2026, 11, 20, 18, tzinfo=UTC).timestamp() * 1000]
    finals: dict[str, str] = {}

    def add(student, experiment, name, role, value, *, params, attrs, study_attrs,
            distributions=None, status="COMPLETE", intermediate=None):  # fmt: skip
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
                        **attrs,
                    },
                    "study": {"direction": "maximize", "attrs": study_attrs},
                    "artifacts": {"design.py": "model = 1\n"}
                    if role == "final" and attrs.get("has_design")
                    else {},
                },
            }
        )
        subs.append(sub)
        return sub.ref()

    def repeats(cfg: dict, machine: float, n: int = 3) -> tuple[float, float]:
        scores = np.array([accuracy(cfg, rng, machine) for _ in range(n)])
        return float(scores.mean()), float(scores.std())

    linear = {"n_hidden_layers": 0, "hidden_dim": 0, "activation": "relu", "norm": "none",
              "skip": False, "dropout": 0.0, "sigma": 0.0, "lr": 1e-3}  # fmt: skip
    own = {
        "alma": ("alma-depth-on-eurosat", "width", 0.012),
        "bram": ("bram-norm-or-skip", "norm", 0.006),
        "chen": ("chen-small-lr-range", "lr", -0.004),
        "dina": ("dina-wide-heads", "width", 0.018),
        "emre": ("emre-sigma-noise", "sigma", 0.0),
        "fenna": ("fenna-pruner-first", "all", 0.009),
    }
    for student in NAMES:
        machine = float(rng.normal(0, 0.004))
        mean, std = repeats(linear, machine)
        # handed-out part
        add(student, f"lesson{lesson}-notebook-runs", "baseline", "baseline", mean,
            params=params_of(linear), attrs={"accuracy_std": std, "n_repeats": 3, "n_params": n_params(linear), "name": "baseline"},
            study_attrs={})  # fmt: skip
        for i in range(24):
            cfg = draw(rng)
            acc = accuracy(cfg, rng, machine)
            p = {k: cfg[k] for k in SPACE}
            add(student, f"lesson{lesson}-notebook", f"trial-{i:02d}", "trial", acc, params=p,
                distributions=SPACE, attrs={"n_params": n_params(cfg), "config": {"dataset": "dtd", **cfg}},
                study_attrs={"search_space": SPACE_LINE, "searched_axes": ",".join(sorted(SPACE)), "origin": "notebook"})  # fmt: skip
        if student not in own:
            continue
        study, focus, edge = own[student]
        space_line = {
            "width": "dataset=dtd | n_hidden_layers ~ int(1, 2) | hidden_dim ~ logint(64, 2048) | lr=0.0005",
            "norm": "dataset=dtd | n_hidden_layers ~ int(1, 3) | norm in {none,batch,layer} | skip in {false,true}",
            "lr": "dataset=dtd | n_hidden_layers=1 | lr ~ logU(3e-4, 3e-3)",
            "sigma": "dataset=dtd | n_hidden_layers=0 | sigma ~ U(0, 0.5)",
            "all": SPACE_LINE,
        }[focus]
        experiment = f"lesson{lesson}-study-{study}"
        sattrs = {
            "search_space": space_line,
            "searched_axes": focus,
            "origin": "study",
            "study": study,
        }
        base_mean, base_std = repeats(linear, machine)
        add(student, f"{experiment}-runs", "baseline", "baseline", base_mean, params=params_of(linear),
            attrs={"accuracy_std": base_std, "n_repeats": 3, "n_params": n_params(linear), "name": "baseline"},
            study_attrs=sattrs)  # fmt: skip
        best_cfg, best_acc = linear, base_mean
        for i in range(int(rng.integers(12, 25))):
            cfg = draw(rng)
            if focus == "lr":
                cfg.update(
                    n_hidden_layers=1,
                    lr=float(math.exp(rng.uniform(math.log(3e-4), math.log(3e-3)))),
                )
            if focus == "sigma":
                cfg.update(n_hidden_layers=0, sigma=float(rng.uniform(0, 0.5)))
            acc = accuracy(cfg, rng, machine) + edge * 0.3
            pruned = acc < base_mean - 0.01 and i >= 5
            p = {k: cfg[k] for k in SPACE}
            add(student, experiment, f"trial-{i:02d}", "trial", acc, params=p, distributions=SPACE,
                status="PRUNED" if pruned else "COMPLETE",
                intermediate=[{"step": s, "value": acc * (1 - 0.3 * math.exp(-s / 2))} for s in range(5 if pruned else 15)],
                attrs={"n_params": n_params(cfg), "config": {"dataset": "dtd", **cfg}}, study_attrs=sattrs)  # fmt: skip
            if not pruned and acc > best_acc:
                best_cfg, best_acc = cfg, acc
        f_mean, f_std = repeats(best_cfg, machine)
        f_mean += edge * 0.3
        add(student, f"{experiment}-runs", "winner", "final", f_mean, params=params_of(best_cfg),
            attrs={"accuracy_std": f_std, "n_repeats": 3, "n_params": n_params(best_cfg), "name": "winner", "has_design": True},
            study_attrs=sattrs)  # fmt: skip
        finals[student] = subs[-1].ref()
    # a few students build on a classmate's final
    for who, target in (("bram", "alma"), ("dina", "alma"), ("hana", "dina")):
        if target in finals:
            cfg = {**linear, "n_hidden_layers": 1, "hidden_dim": 256, "lr": 5e-4}
            mean, std = repeats(cfg, 0.0)
            add(who, f"lesson{lesson}-study-{who}-builds", "builds", "hypothesis", mean, params=params_of(cfg),
                attrs={"accuracy_std": std, "n_repeats": 3, "n_params": n_params(cfg), "builds_on": finals[target], "name": "builds"},
                study_attrs={"origin": "study", "study": f"{who}-builds"})  # fmt: skip
    return subs
