"""The study of an exercise, the same in every lesson: a baseline, a search over a design
of the student's own, the winner rerun, and the difference in sigmas.

A lesson says what is its own in a `Harness` (`hpo.harness`). Every lesson's
`exercises/study.py` is then the student's answers (`Questions`), a space, a design, and
one call:

    Exercise(HARNESS, QUESTIONS, SPACE, DESIGN, make_model, Path(designs.__file__)).run()

The search goes to the study `lesson<N>-study-<study>`, the baseline and the winner to
`lesson<N>-study-<study>-runs`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
from notebooktester import param
from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn

from hpo.architecture import Architecture
from hpo.budget import Budget
from hpo.compare import Comparison
from hpo.designs import Design
from hpo.harness import Baseline, Data, Harness
from hpo.ideas import Idea, study_name
from hpo.metrics import Objective, Repeats
from hpo.runs import DesignScript, RunRecord, StudyLog
from hpo.space import Choice, Config, Space
from hpo.store import Store, under_tester
from hpo.training import Fit, repeat

EXPECTS_PLACEHOLDER = "say what you expect, and why"

# builds the model a drawn config chooses, on the data of that config
MakeModel = Callable[[Config, Data], nn.Module]


class Questions(BaseModel):
    """What the student answers at the top of `study.py`, before anything runs."""

    model_config = ConfigDict(frozen=True)

    # your first name and what the study studies: "sam-gate-on-eurosat"
    study: str
    # what it investigates, and the idea of the class it builds on
    idea: Idea
    # what you expect to happen, and why: one line, written before you run
    expects: str
    # the metric the search maximises; every run logs them all
    objective: Objective = "accuracy"
    # the time you give the study on this machine
    minutes: float

    @field_validator("study")
    @classmethod
    def _named(cls, name: str) -> str:
        return study_name(name)

    @field_validator("expects")
    @classmethod
    def _said(cls, expects: str) -> str:
        said = expects.strip()
        if not said or (said == EXPECTS_PLACEHOLDER and not under_tester()):
            raise ValueError(
                "EXPECTS: say in one line what you expect to happen and why, before "
                f"you run, instead of {EXPECTS_PLACEHOLDER!r}"
            )
        return said


class Outcome(BaseModel):
    """What a study found, as data."""

    study: str
    objective: Objective
    budget: Budget
    best: Config
    baseline: Comparison
    winner: Comparison


class Exercise:
    """One study of a student, over `design` with `space`, in a lesson's harness.

    The cost of a trial is measured on the search study `like`, `lesson<N>-notebook` by
    default: the handed-out search of the lesson's exercise notebook."""

    def __init__(
        self,
        harness: Harness,
        questions: Questions,
        space: Space,
        design: Design,
        make_model: MakeModel,
        design_file: Path,
        *,
        pruner: optuna.pruners.BasePruner | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        like: str | None = None,
        store: Store | None = None,
    ) -> None:
        self.harness = harness
        self.questions = questions
        self.space = space
        self.design = design
        self.make_model = make_model
        self.script = DesignScript.read(design_file)
        self.pruner = pruner
        self.sampler = sampler
        self.store = store or Store.default()
        self.like = like or f"lesson{harness.lesson}-notebook"
        # The tester's run of a script keeps a study of its own: under notebooktester
        # the placeholder names the notebook's own study as well, which holds another
        # space.
        testing = "-script" if self.store.testing else ""
        self.name = f"lesson{harness.lesson}-study-{questions.study}{testing}"

    @property
    def objective(self) -> Objective:
        return self.questions.objective

    def budget(self) -> Budget:
        try:
            like = StudyLog.load(self.like, self.store)
            return Budget.measure(
                self.questions.minutes,
                like,
                space=self.space,
                epochs=self.harness.epochs,
                repeats=self.harness.repeats,
            )
        except ValueError as error:
            raise ValueError(
                f"budget: run Part 1 of lesson {self.harness.lesson}'s exercise notebook "
                f"on this machine first; it measures what one trial costs here ({error})"
            ) from None

    def repeats(
        self, make: Callable[[], nn.Module], data: Data, config: Config
    ) -> Repeats:
        return repeat(
            make,
            lambda model: self.harness.fit(model, data, config),
            self.harness.repeats,
        )

    def baselines(self) -> dict[str, tuple[Baseline, Data, Repeats]]:
        """The baseline of every dataset the space can draw, measured before the search."""
        drawn = self.space.get("dataset")
        datasets = drawn.choices if isinstance(drawn, Choice) else [drawn]
        out = {}
        for dataset in map(str, datasets):
            base = self.harness.baseline(dataset)
            data = self.harness.data(base.config)
            make = base.make
            out[dataset] = (
                base,
                data,
                self.repeats(
                    lambda make=make, data=data: make(data), data, base.config
                ),
            )
            print(f"baseline : measured on {dataset}")
        return out

    def trial(self, search: StudyLog) -> Callable[[optuna.Trial], float]:
        """What the search calls once per trial: draw a config, build it, train it with
        the harness, report the objective after every epoch, and return it after the
        last. A pruned trial is recorded as well, with the epochs it ran."""

        def run_trial(trial: optuna.Trial) -> float:
            config = self.space.draw(trial)
            data = self.harness.data(config)
            model = self.make_model(config, data)
            seen = Fit()

            # every epoch reaches the store, and the study's pruner can stop a trial
            # that is behind the others
            def to_optuna(epoch: int, fit: Fit) -> None:
                nonlocal seen
                seen = fit
                trial.report(fit.scores().get(self.objective), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                seen = self.harness.fit(model, data, config, to_optuna)
                return seen.scores().get(self.objective)
            finally:
                search.record(
                    trial,
                    RunRecord(
                        role="trial",
                        dataset=str(config.get("dataset")),
                        config=config,
                        fixed=self.space.fixed(),
                        train_seconds=seen.seconds,
                        curves=seen.curves,
                        scores=seen.confusion.scores() if seen.confusion else None,
                        confusion=seen.confusion,
                        extra=dict(data.dims),
                    ),
                    Architecture.of(model, data.example),
                    self.script,
                )

        return run_trial

    def run(self) -> Outcome:
        """Check the space, turn the minutes into trials, measure the baselines, search,
        rerun the winner, log baseline and winner, and print what it found."""
        design = self.design
        self.space.check(design.options(), design.choices(), self.harness.trainer)
        budget = self.budget()
        n_trials = param(budget.n_trials, test=2)
        print(f"budget   : {budget}")
        if n_trials == 0:
            raise ValueError(
                f"budget: {budget.minutes} minutes hold no trial on this machine once the "
                "baseline and the winner are set aside; give the study more minutes or "
                "fewer epochs"
            )
        baselines = self.baselines()

        idea = self.questions.idea
        search = StudyLog.search(
            self.name,
            self.space,
            idea,
            self.objective,
            sampler=self.sampler,
            pruner=self.pruner,
            attrs={
                "origin": "study",
                "study": self.questions.study,
                "budget": {**budget.model_dump(), "n_trials": n_trials},
                # a study over a design of your own never searches the handed-out space
                "space_is_default": False,
                "expects": self.questions.expects,
            },
            store=self.store,
        )
        start = time.time()
        search.optimize(self.trial(search), n_trials)
        done = search.trials(optuna.trial.TrialState.COMPLETE)
        table = pd.DataFrame([{self.objective: t.value, **t.params} for t in done])
        print(
            f"{n_trials} trials in {time.time() - start:.0f}s; {len(done)} finished in all"
        )
        print(
            table.sort_values(self.objective, ascending=False)
            .head(6)
            .to_string(index=False)
        )

        best = search.best()
        runs = StudyLog.runs(
            f"{self.name}-runs", idea, self.objective,
            choices=self.harness.choices, store=self.store,
        )  # fmt: skip
        base, base_data, base_runs = baselines[str(best.get("dataset"))]
        runs.log(
            "baseline",
            "baseline",
            base_runs,
            params=self.harness.params(base.config, base_data),
            dataset=str(base.config.get("dataset")),
            architecture=Architecture.of(base.make(base_data), base_data.example),
            design=self.script,
        )
        data = self.harness.data(best)
        win_runs = self.repeats(lambda: self.make_model(best, data), data, best)
        runs.log(
            "winner",
            "final",
            win_runs,
            params=self.harness.params(best, data),
            dataset=str(best.get("dataset")),
            architecture=Architecture.of(self.make_model(best, data), data.example),
            design=self.script,
            attrs={"search_space": self.space.line()},
        )

        baseline = Comparison.of(
            f"baseline on {best.get('dataset')}", base_runs.values(self.objective)
        )
        winner = Comparison.of(
            "your search winner",
            win_runs.values(self.objective),
            versus=base_runs.values(self.objective),
        )
        print()
        print("winner   :", {k: v for k, v in best.items() if k != "epochs"})
        print(f"objective: {self.objective}")
        print(baseline)
        print(winner)
        print(f"\nthe winner against the baseline: {winner.sigmas:+.1f} sigma")
        print(f"idea     : {idea}")
        print(f"studies  : {self.name}, {self.name}-runs  (make optuna-dashboard)")
        return Outcome(
            study=self.name,
            objective=self.objective,
            budget=budget,
            best=best,
            baseline=baseline,
            winner=winner,
        )
