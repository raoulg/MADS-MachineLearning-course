"""A run as the class run protocol writes it, and the study it is written to.

Optuna stores studies; a study holds trials; a trial holds the params it drew, one value
per epoch, its final value and free-form user attrs. `StudyLog` is the one class that
writes on Optuna's studies and trials, in the shape the protocol asks:

- a search is a study (`StudyLog.search`): every trial draws its params from the search
  space and reports its objective per epoch, so a pruner can stop it early; `record`
  writes what the trial was;
- a run that is not a draw from a space (a baseline, a control, a winner rerun) is a
  trial in a second study (`StudyLog.runs` + `log`), with the params it was given and
  one value per repeat.

A study names its objective, the metric of `hpo.measure.metrics.Objective` that is the value of
its trials and that its search maximises, and carries the idea it investigates. It holds
one idea, the trials of one space and one objective, and refuses another.

Every trial carries, as user attrs, what the protocol asks of a run (`RunRecord`): its
`run_id`, `role`, `session`, `dataset`, `train_seconds`, curves, every metric
(`metrics`, `metrics_std`), the confusion over its repeats (`confusion`), and with an
architecture the model's `n_params`, its `flops_per_image` and `latency_us` (the cost of
one image's forward pass) and its one-line `architecture`. The layer table goes
along as the trial's note and the artifact `architecture.json`, the design script as
the artifact `design.py`.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
from optuna.artifacts import FileSystemArtifactStore, upload_artifact
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import FrozenTrial, TrialState
from optuna_dashboard import save_note
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hpo.measure.architecture import Architecture
from hpo.measure.metrics import (
    OBJECTIVES,
    Confusion,
    Objective,
    Repeats,
    Results,
    Scores,
)
from hpo.measure.training import Curves
from hpo.record.ideas import Idea
from hpo.record.protocol import MAX_ARTIFACT_BYTES, SUBMISSION_REF
from hpo.record.store import Store
from hpo.search.space import Config, Param, Space

# set_metric_names and the artifact store are marked experimental in Optuna
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# What a run is for. The class view compares baselines with finals per student and reads
# trials as the search space everyone explored.
Role = Literal["baseline", "trial", "final", "control", "hypothesis"]

# One timestamp per execution of a notebook (this module is imported once, at the top):
# every trial carries it as the user attr `session`, so a notebook run twice can be told
# apart and the class view keeps only a student's latest session of the handed-out runs.
SESSION = time.strftime("%Y-%m-%d %H:%M:%S")

# What a user attr can hold: what Optuna stores as JSON.
type Json = str | int | float | bool | None | list[Json] | dict[str, Json]


def _refs(value: str | Sequence[str] | None) -> list[str]:
    """The submission_refs in `value` (one string, comma-separated, or a list), stripped;
    a `ValueError` names one that is not a submission_ref."""
    parts = [value] if isinstance(value, str) else list(value or [])
    refs = [ref.strip() for part in parts for ref in part.split(",") if ref.strip()]
    for ref in refs:
        if not SUBMISSION_REF.fullmatch(ref):
            raise ValueError(
                f"builds_on: {ref!r} is not a submission_ref lesson<N>/<login>/<run_id>; "
                "copy the user attr submission_ref from the classmate's run in the class view"
            )
    return refs


class RunRecord(BaseModel):
    """What a trial carries as user attrs (`attrs`)."""

    model_config = ConfigDict(frozen=True)

    role: Role
    session: str = SESSION
    name: str | None = None
    dataset: str | None = None
    train_seconds: float | None = None
    curves: Curves | None = None
    # the config a trial drew, and the values its space fixes
    config: Config | None = None
    fixed: Config = Field(default_factory=dict)
    # every metric: the mean over the repeats and the spread, the confusion of them all
    scores: Scores | None = None
    spread: Scores | None = None
    confusion: Confusion | None = None
    # per scalar metric the spread over the repeats, under `<metric>_std`
    scalar_spread: dict[str, float] = Field(default_factory=dict)
    n_repeats: int | None = None
    # the classmates' runs this one builds on, by their submission_ref
    builds_on: list[str] = Field(default_factory=list)
    extra: dict[str, Param] = Field(default_factory=dict)

    @field_validator("builds_on", mode="before")
    @classmethod
    def _builds_on(cls, value: str | Sequence[str] | None) -> list[str]:
        return _refs(value)

    @classmethod
    def of_results(cls, role: Role, results: Results, **fields: object) -> RunRecord:
        """The record of a run that is given its params, from what its repeats logged."""
        logged = {m: results.values(m) for m in results.logged()}
        every = (
            {
                "scores": results.mean(),
                "spread": results.std(),
                "confusion": results.confusion(),
            }
            if isinstance(results, Repeats)
            else {}
        )
        return cls.model_validate(
            {
                "role": role,
                "train_seconds": results.seconds,
                "scalar_spread": {m: float(v.std()) for m, v in logged.items()},
                "n_repeats": len(results),
                **every,
                **fields,
            }
        )

    def attrs(self) -> dict[str, Json]:
        """The user attrs, under the names the class run protocol reads."""
        out: dict[str, Json] = {"role": self.role, "session": self.session}
        if self.name is not None:
            out["name"] = self.name
        if self.dataset is not None:
            out["dataset"] = self.dataset
        if self.train_seconds is not None:
            out["train_seconds"] = round(self.train_seconds, 3)
        for name, values in (self.curves.attrs() if self.curves else {}).items():
            curve: list[Json] = list(values)
            out[name] = curve
        if self.config is not None:
            out["config"] = dict(self.config)
        out.update(self.fixed)
        if self.scores is not None:
            out["metrics"] = self.scores.rounded().model_dump()
        if self.spread is not None:
            out["metrics_std"] = self.spread.rounded().model_dump()
        if self.confusion is not None:
            out["confusion"] = self.confusion.model_dump()
        out.update({f"{m}_std": round(v, 5) for m, v in self.scalar_spread.items()})
        if self.n_repeats is not None:
            out["n_repeats"] = self.n_repeats
        if self.builds_on:
            out["builds_on"] = ",".join(self.builds_on)
        out.update(self.extra)
        return out


class DesignScript(BaseModel):
    """The script that defines a run's architectures, carried as the artifact
    `design.py`."""

    model_config = ConfigDict(frozen=True)

    text: str

    @classmethod
    def read(cls, path: str | Path) -> DesignScript:
        """The text of `path`; a `ValueError` when it is larger than a submission
        carries (`MAX_ARTIFACT_BYTES`, measured as JSON)."""
        text = Path(path).read_text(encoding="utf-8")
        size = len(json.dumps(text))
        if size > MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"design: {path} is {size} bytes as JSON, more than a submission "
                f"carries ({MAX_ARTIFACT_BYTES}); keep only the architectures in it"
            )
        return cls(text=text)


def _distribution(
    name: str, value: Param, choices: Mapping[str, Sequence[str]]
) -> BaseDistribution | None:
    """The distribution a given param is stored under, or None for a value Optuna cannot
    hold as a param (a string without an entry in `choices`)."""
    if isinstance(value, bool | np.bool_):
        return CategoricalDistribution([False, True])
    if isinstance(value, int | np.integer):
        return IntDistribution(int(value), int(value))
    if isinstance(value, float | np.floating):
        return FloatDistribution(float(value), float(value))
    if name in choices and value in choices[name]:
        return CategoricalDistribution(list(choices[name]))
    return None


class StudyLog:
    """One study in the store, and every run written to it.

    `choices` lists, per param name, the values a string param can take: Optuna holds a
    categorical param to one list of choices for the whole study, so a string param of
    `log` needs one. A string param without an entry is written as a user attr."""

    def __init__(
        self,
        study: optuna.Study,
        store: Store,
        choices: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.study = study
        self.store = store
        self.choices = dict(choices or {})

    @property
    def name(self) -> str:
        return self.study.study_name

    @property
    def objective(self) -> Objective:
        held = self.study.user_attrs.get("objective", "accuracy")
        return held if held in OBJECTIVES else "accuracy"

    @classmethod
    def search(
        cls,
        name: str,
        space: Space,
        idea: Idea | None = None,
        objective: Objective = "accuracy",
        *,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        attrs: Mapping[str, Json] | None = None,
        store: Store | None = None,
    ) -> StudyLog:
        """The search study `name`, made if it does not exist yet, that maximises
        `objective` over `space`. `sampler=None` draws every trial at random and
        independently; `pruner=None` lets every trial run to its last epoch. The study
        carries its space (`search_space`, `searched_axes`), `idea`, `objective` and
        whatever `attrs` holds; a study that exists already gets the new trials."""
        log = cls._open(
            store or Store.default(),
            name,
            idea,
            objective,
            space=space,
            sampler=sampler or optuna.samplers.RandomSampler(),
            pruner=pruner or optuna.pruners.NopPruner(),
        )
        log.study.set_user_attr("search_space", space.line())
        log.study.set_user_attr("searched_axes", ",".join(space.searched()))
        for key, value in (attrs or {}).items():
            log.study.set_user_attr(key, value)
        return log

    @classmethod
    def runs(
        cls,
        name: str,
        idea: Idea | None = None,
        objective: Objective = "accuracy",
        *,
        choices: Mapping[str, Sequence[str]] | None = None,
        store: Store | None = None,
    ) -> StudyLog:
        """The study `name` of the runs that are given their params, made if it does not
        exist yet. The value of its trials is `objective`."""
        return cls._open(
            store or Store.default(), name, idea, objective, choices=choices
        )

    @classmethod
    def load(cls, name: str, store: Store | None = None) -> StudyLog:
        """The study `name` as it is in the store; a `ValueError` when it is not there."""
        store = store or Store.default()
        if name not in optuna.get_all_study_names(store.url):
            raise ValueError(f"no study {name!r} in the store; run it first")
        return cls(optuna.load_study(study_name=name, storage=store.url), store)

    @classmethod
    def _open(
        cls,
        store: Store,
        name: str,
        idea: Idea | None,
        objective: Objective,
        *,
        space: Space | None = None,
        choices: Mapping[str, Sequence[str]] | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> StudyLog:
        """A study that holds another idea, trials of another space or of another
        objective raises a `ValueError`; under notebooktester it is started again, since
        the tester's store only says whether the code runs."""
        if objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES}, got {objective!r}"
            )

        def create() -> optuna.Study:
            return optuna.create_study(
                study_name=name,
                storage=store.url,
                direction="maximize",
                load_if_exists=True,
                sampler=sampler,
                pruner=pruner,
            )

        log = cls(create(), store, choices)
        problem = log.refuses(idea, space, objective)
        if problem is not None:
            if not store.testing:
                raise ValueError(problem)
            optuna.delete_study(study_name=name, storage=store.url)
            log = cls(create(), store, choices)
        log.study.set_metric_names([objective])
        log.study.set_user_attr("objective", objective)
        for key, value in (idea.attrs() if idea else {}).items():
            log.study.set_user_attr(key, value)
        return log

    def refuses(
        self, idea: Idea | None, space: Space | None, objective: Objective
    ) -> str | None:
        """Why this study cannot take `idea`, `space` and `objective`, or None."""
        attrs, name = self.study.user_attrs, self.name
        held_objective = attrs.get("objective", "accuracy")
        if held_objective != objective and self.study.trials:
            return (
                f"the study {name!r} holds trials on the objective {held_objective!r}, "
                f"not {objective!r}: values on two metrics cannot be read on one axis, "
                "so give the study with this objective another name"
            )
        held = attrs.get("idea_id")
        if idea is not None and held is not None and held != idea.id:
            return (
                f"the study {name!r} holds the idea {attrs['idea_name']} "
                f"({attrs['idea']!r}), not {idea}: another idea is another study, so "
                "give this one another name; name the old idea in builds_on if the new "
                "one builds on it"
            )
        searched = attrs.get("search_space")
        if (
            space is not None
            and searched not in (None, space.line())
            and self.study.trials
        ):
            return (
                f"the study {name!r} holds trials of another search space:\n"
                f"  held : {searched}\n  given: {space.line()}\n"
                "trials of two spaces in one study cannot be read together, so give the "
                "study with this space another name, or put the space back as it was"
            )
        return None

    def record(
        self,
        trial: optuna.Trial,
        record: RunRecord,
        architecture: Architecture | None = None,
        design: DesignScript | None = None,
    ) -> None:
        """Write `record` on `trial` as its user attrs, with a `run_id` made once; the
        architecture becomes the trial's note and, with the design script, its
        artifacts."""
        if "run_id" not in trial.user_attrs:
            trial.set_user_attr("run_id", uuid.uuid4().hex)
        for key, value in record.attrs().items():
            trial.set_user_attr(key, value)
        artifacts: dict[str, str] = {}
        if architecture is not None:
            trial.set_user_attr("n_params", architecture.n_params)
            trial.set_user_attr("flops_per_image", architecture.flops_per_image)
            trial.set_user_attr("latency_us", architecture.latency_us)
            trial.set_user_attr("architecture", architecture.signature)
            save_note(trial, architecture.note())
            artifacts["architecture.json"] = architecture.model_dump_json(indent=1)
        if design is not None:
            artifacts["design.py"] = design.text
        if artifacts:
            self._upload(trial, artifacts)

    def _upload(self, trial: optuna.Trial, artifacts: Mapping[str, str]) -> None:
        self.store.artifacts.mkdir(parents=True, exist_ok=True)
        target = FileSystemArtifactStore(str(self.store.artifacts))
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in artifacts.items():
                path = Path(tmp) / name
                path.write_text(text, encoding="utf-8")
                upload_artifact(
                    artifact_store=target, file_path=str(path), study_or_trial=trial
                )

    def log(
        self,
        name: str,
        role: Role,
        results: Results,
        *,
        params: Mapping[str, Param] | None = None,
        dataset: str | None = None,
        architecture: Architecture | None = None,
        design: DesignScript | None = None,
        builds_on: str | Sequence[str] | None = None,
        attrs: Mapping[str, Param] | None = None,
    ) -> int:
        """Write one run that is not a draw from a search space as a trial, and return
        its number.

        - `params`: the choices the run was given, each under a stable name. They are the
          trial's params, so the dashboard can put them on an axis.
        - `results`: what its repeats logged. The mean of the study's objective is the
          trial's value, every repeat an intermediate value (step = repeat).
        - `builds_on`: the classmates' runs it builds on, by their `submission_ref`."""
        if self.objective not in results.logged():
            raise ValueError(
                f"log: the study {self.name!r} maximises {self.objective!r}, which this "
                f"run does not log (it logs {results.logged()})"
            )
        given: dict[str, Param] = {}
        distributions: dict[str, BaseDistribution] = {}
        loose: dict[str, Param] = {}
        for key, value in (params or {}).items():
            dist = _distribution(key, value, self.choices)
            if dist is None:
                loose[key] = value
            else:
                given[key] = value.item() if isinstance(value, np.generic) else value
                distributions[key] = dist
        # a record that cannot be written raises before a trial exists
        record = RunRecord.of_results(
            role,
            results,
            name=name,
            dataset=dataset,
            builds_on=builds_on,
            extra={**loose, **(attrs or {})},
        )
        values = results.values(self.objective)
        self.study.enqueue_trial(given)
        trial = self.study.ask(distributions)
        for step, value in enumerate(values):
            trial.report(float(value), step)
        self.record(trial, record, architecture, design)
        self.study.tell(trial, float(values.mean()))
        return trial.number

    def optimize(
        self, objective: Callable[[optuna.Trial], float], n_trials: int
    ) -> None:
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    def trials(self, *states: TrialState) -> list[FrozenTrial]:
        """The trials in `states`, every trial without."""
        return [t for t in self.study.trials if not states or t.state in states]

    def best(self) -> Config:
        """The config of the best trial."""
        return dict(self.study.best_trial.user_attrs["config"])

    def seconds_per_trial(self, epochs: int | None = None) -> float:
        """The median seconds a finished trial took, from start to end, scaled to
        `epochs` when the trials say how many epochs they ran."""
        done = self.trials(TrialState.COMPLETE)
        took = [t.duration.total_seconds() for t in done if t.duration is not None]
        if not took:
            raise ValueError(f"the study {self.name!r} has no finished trial yet")
        seconds = statistics.median(took)
        ran = [t.user_attrs.get("config", {}).get("epochs") for t in done]
        ran = [e for e in ran if isinstance(e, int) and e > 0]
        if epochs is not None and ran:
            seconds *= epochs / statistics.median(ran)
        return seconds
