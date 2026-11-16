"""The run submission format, and the validator for a directory of submissions.

One submitted run is one Optuna trial, as one JSON file in the class runs repository, at

    lesson<N>/<student>/<run_id>.json

where <student> is the student's GitHub login and <run_id> the trial's user attr
`run_id`, 32 hex characters made when the trial was recorded. The file holds the trial as
Optuna holds it: its params with the distribution each was drawn from, its final value
and the value after every epoch, its user attrs (role, dataset, architecture, ...), the
user attrs of its study (the search space), small text/JSON artifacts (the architecture,
the design script `design.py`), and where the code came from. Schema `mads-run/2`.

A run logs every metric of `hpo.measure.metrics` in the user attrs `metrics` (the value after the
last epoch, or the mean over the repeats) and `metrics_std` (the spread over the
repeats), and the confusion matrix over all its repeats in `confusion`, from which a
metric that counts can be derived later. Its study names its objective, the metric that is the trial's value, in the
study attr `objective`. A run without `metrics` logs its value only, and a study without
`objective` is on accuracy.

This module depends on pydantic and the standard library only, so the runs repository's
CI can validate a push or a pull request without installing Optuna:

    uv run --with pydantic python scripts/run_submission.py check <runs-repo> \
        [--actor <login>] [--teachers raoulg] \
        [--files <newline-separated paths>... | --files-from <NUL-separated file>]

With `--files` or `--files-from`, `<runs-repo>` is the repository as it would be with
the change applied, and the paths are every path the change touches, deleted ones
included.

`docs/run-protocol.md` describes the protocol around it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

SCHEMA = "mads-run/2"
MAX_FILE_BYTES = 2_000_000
MAX_ARTIFACT_BYTES = 256_000
MAX_ATTR_BYTES = 20_000  # one user attr, measured as JSON
MAX_CHANGED_FILES = 200
ARTIFACT_SUFFIXES = (".json", ".txt", ".md", ".yaml", ".yml", ".py")
# Every pattern is used with fullmatch, so a trailing newline never slips through.
LOGIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", re.ASCII)  # GitHub's login rule
RUN_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)
PATH = re.compile(r"lesson(\d+)/([A-Za-z0-9-]+)/([0-9a-f]{32})\.json", re.ASCII)
# A run another one builds on, as the user attr `builds_on` names it: a submission's path
# without `.json`. Login and run id follow LOGIN and RUN_ID.
SUBMISSION_REF = re.compile(
    r"lesson\d+/[A-Za-z0-9][A-Za-z0-9-]{0,38}/[0-9a-f]{32}", re.ASCII
)
EXPERIMENT = re.compile(r"lesson\d+-[a-z0-9-]+", re.ASCII)
KEY = re.compile(r"[\w\-. /]{1,250}", re.ASCII)  # a param or user attr name
ARTIFACT_PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", re.ASCII)
# What sqlite holds as an integer; a larger step or time cannot be stored.
MAX_INT = 2**63 - 1
Scalar = str | int | float | bool
# The attributes each distribution kind carries, as Optuna writes them.
DISTRIBUTIONS = {
    "FloatDistribution": {"low", "high", "log", "step"},
    "IntDistribution": {"low", "high", "log", "step"},
    "CategoricalDistribution": {"choices"},
}


# The metrics a run logs, as hpo.measure.metrics computes them: one value for the whole test set,
# and one value per class. A study's objective is one of the scalars.
SCALAR_METRICS = ("accuracy", "macro_recall")
PER_CLASS_METRICS = ("recall", "precision", "f1")
OBJECTIVES = SCALAR_METRICS
DEFAULT_OBJECTIVE = "accuracy"
CLASS_NAME = re.compile(r"[\w\-. ]{1,64}", re.ASCII)
# How far a trial's value may lie from its objective in `metrics`: both are rounded.
VALUE_TOLERANCE = 1e-4


# An idea: what a study investigates, in a few words. Its id is a hash of its author,
# its words and the names of the ideas it builds on, so an id holds its parents the way
# a git commit's hash holds its parents'; its name is that id made readable. The five
# study attrs below carry it.
IDEA_ATTRS = ("idea", "idea_author", "idea_parents", "idea_id", "idea_name")
IDEA_NAME = re.compile(r"[a-z]+(?:-[a-z]+)*-[0-9]{2}", re.ASCII)
MAX_IDEA_NAME_CHARS = 60
MIN_IDEA_CHARS, MAX_IDEA_CHARS = 8, 120
# The author of what a lesson hands out to everyone: one idea the whole class shares.
COURSE = "course"
HANDED_OUT = re.compile(r"lesson\d+-notebook[a-z0-9-]*", re.ASCII)

# 256 words each, so one byte of an id picks one.
ADJECTIVES = """
    animated facetious stippled peppy joking riotous spotted multicolored vivid
    opalescent vigorous malleable translucent gamesome showy tittering variegated
    far-out agile glittery giggling hooting blissful bubbly checkered comical
    speckled groovy splattered gossamer foamy carbonated jocular dazzling ornate
    hilarious freckled deep harlequin loony velvety fluffy lighthearted laughter
    lively silky glowing outlandish laughing fizzy wacky frisky ecstatic bizarre
    effervescent rollicking humorous intense cheerful elated ludicrous kaleidoscopic
    absurd jovial witty slapstick lustrous zany ridiculous luminous spellbinding
    patchwork convivial flashy dynamic frothy piebald mottled captivating snappy
    effulgent spirited bedazzling shiny colorful exuberant cackling entertaining
    diaphanous teasing farcical spangled sudsy prankish light decorated vivacious
    happy speedy goofy good-natured furry pied crystalline sportive ostentatious
    gut-busting snickering bejeweled eye-catching chortling cordial jocose brilliant
    extravagant prismatic breezy chuckling iridescent limpid perky ebullient jolly
    affable incandescent bantering waggish mirthful smiley bellowing joyful sunny
    giggly brisk pliable quick glossy striking mesmerizing billowy roaring blithe
    zesty sprightly beaming mischievous sidesplitting mind-bending enchanting
    whimsical kidding bright upbeat sparkling sleek droll quirky uproarious flecked
    shimmery pearlescent coltish amusing plaid transparent vibrant plush energetic
    supple gaudy wispy mind-boggling shining frolicsome lithe smooth coruscating
    mind-expanding giddy kooky rubbery snorting nimble airy knee-slapping chipper
    embellished adorned soft clear polished downy jesting trippy bewitching bouncy
    good-humored playful bold hysterical gleaming eccentric grinning jubilant
    carefree motley ethereal rainbow-hued amiable psychedelic resplendent spattered
    scintillating jaunty dotted zippy peculiar merry howling buoyant flexible
    sparkly dappled cushy polychromatic twinkling gleeful roguish saturated squishy
    elastic genial springy flamboyant puckish impish elaborate swift delightful
    funny silly nutty pellucid hypnotic odd mind-blowing cheery glittering diverting
    rich limber radiant feathery fuzzy festooned rib-tickling
""".split()  # noqa: SIM905
ANIMALS = """
    quetzal koala spider mongoose fish mouse rat chinchilla zebra mule camel owl
    cormorant lobster echidna mandrill alpaca curlew mink gorilla panther cat goose
    aardwolf beaver duck lark caterpillar mallard trout vulture tapir goshawk
    salamander hippocampus chicken flamingo quail pheasant hyena vicuna monkey toad
    kudu finch ostrich partridge skunk guanaco gull hedgehog kingfisher sparrow bat
    rhinoceros red-eyed-tree-frog crocodile bee porcupine frog weasel aardvark
    ferret lemur elk kiwi dhole gibbon falcon ibex shrew hornet dolphin quokka
    chamois dugong fossa porpoise bouncing raccoon caribou sheep lyrebird kouprey
    goldfish zebu gnu grasshopper hummingbird dove dog cheetah spoonbill seal
    albatross oyster capybara deer pigeon macaroni-penguin chimpanzee opossum
    elephant swallow bear snail wren iguana crow hare mosquito rabbit rook dingo
    locust parrot sand-dollar yapok boar reindeer woodcock alligator jaguar cattle
    jellyfish emu hawk yak dinosaur turkey dotterel herring stork ape mole magpie
    rail numbat chough sandpiper lapwing cod platypus cobra newt tiger horse jackal
    jay loris kangaroo cassowary ibis ram stingray antelope wolf woodpecker clam
    hamster manatee red-deer seahorse wolverine wasp jerboa serval coati
    whistling-duck moose ermine human wallaby xenops crab pony whale marten stinkbug
    grouse narwhal nightingale bison goldfinch giraffe salmon turtle uakari
    red-panda ocelot louse worm walrus gerbil octopus termite tardigrade snake
    tarsier lion dunlin pig quelea oryx swan chameleon viper butterfly kookabura
    badger eel meerkat dogfish baboon scorpion penguin wombat crane llama sardine
    shark squirrel ant wildcat raven binturong squid anteater otter dragonfly
    armadillo coyote leopard buffalo pelican fennec-fox okapi gazelle eland fox
    cockroach heron barracuda goat beluga eagle axolotl donkey lemming peafowl
    starling hippopotamus sloth fly gnat gaur
""".split()  # noqa: SIM905


def idea_words(text: str) -> str:
    """The words of an idea as they are hashed and stored: lowercase, single spaces."""
    return " ".join(text.lower().split())


def idea_parents(value: Any) -> list[str]:
    """The names in the study attr `idea_parents`, comma-separated; [] when empty."""
    return [] if not isinstance(value, str) or not value else value.split(",")


def idea_id(author: str, idea: str, parents: list[str]) -> str:
    """The id of an idea: sha256 over its author, its words and its parents' names."""
    body = {
        "author": author.lower(),
        "idea": idea_words(idea),
        "parents": sorted(parents),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def idea_name(id_: str) -> str:
    """An idea's id as a name to say and to type: `<adjective>-<animal>-<NN>`, read from
    the first three bytes of the id."""
    first, second, third = bytes.fromhex(id_[:6])
    return f"{ADJECTIVES[first]}-{ANIMALS[second]}-{third % 100:02d}"


def check_idea(attrs: dict[str, Any]) -> None:
    """A study's idea attrs, when it carries any: all five, and id and name are what the
    words, the author and the parents give."""
    present = [key for key in IDEA_ATTRS if key in attrs]
    if not present:
        return
    missing = sorted(set(IDEA_ATTRS) - set(present))
    if missing:
        raise ValueError(f"an idea carries all of {IDEA_ATTRS}; missing {missing}")
    if not all(isinstance(attrs[key], str) for key in IDEA_ATTRS):
        raise ValueError(f"the study attrs {IDEA_ATTRS} are text")
    words, author = attrs["idea"], attrs["idea_author"]
    if words != idea_words(words) or not (
        MIN_IDEA_CHARS <= len(words) <= MAX_IDEA_CHARS
    ):
        raise ValueError(
            f"study attr 'idea': {MIN_IDEA_CHARS} to {MAX_IDEA_CHARS} characters, "
            "in lowercase with single spaces"
        )
    if author != author.lower() or not LOGIN.fullmatch(author):
        raise ValueError("study attr 'idea_author' is a GitHub login in lowercase")
    parents = idea_parents(attrs["idea_parents"])
    for parent in parents:
        if len(parent) > MAX_IDEA_NAME_CHARS or not IDEA_NAME.fullmatch(parent):
            raise ValueError(
                f"study attr 'idea_parents': {parent[:MAX_IDEA_NAME_CHARS]!r} is not "
                "the name of an idea, like blissful-bee-18"
            )
    if parents != sorted(set(parents)):
        raise ValueError("study attr 'idea_parents': sorted, each name once")
    expected = idea_id(author, words, parents)
    if attrs["idea_id"] != expected:
        raise ValueError(
            "study attr 'idea_id' is not the id of this idea, its author and its "
            "parents: submit again, do not edit by hand"
        )
    if attrs["idea_name"] != idea_name(expected):
        raise ValueError(
            "study attr 'idea_name' is not the name of this idea's id: submit again, "
            "do not edit by hand"
        )
    if attrs["idea_name"] in parents:
        raise ValueError("study attr 'idea_parents' names this idea itself")


def check_metrics(key: str, metrics: Any, spread: bool = False) -> None:
    """The user attr `metrics` (or, with `spread`, `metrics_std`): the scalar metrics as
    numbers, the metrics per class as a mapping of class name to number; a metric is
    between 0 and 1, a spread at least 0."""
    if not isinstance(metrics, dict):
        raise ValueError(f"attr {key!r} is a mapping of metric to value")  # noqa: TRY004
    known = (*SCALAR_METRICS, *PER_CLASS_METRICS)
    unknown = sorted(set(metrics) - set(known))
    if unknown:
        raise ValueError(f"attr {key!r}: unknown metrics {unknown}; known are {known}")

    def number(name: str, value: Any) -> None:
        if not is_number(value) or value < 0 or (not spread and value > 1):
            bound = "at least 0" if spread else "between 0 and 1"
            raise ValueError(f"attr {key!r}: {name} is a number {bound}")

    for name in SCALAR_METRICS:
        if name in metrics:
            number(name, metrics[name])
    for name in PER_CLASS_METRICS:
        if name not in metrics:
            continue
        per_class = metrics[name]
        if not isinstance(per_class, dict):
            raise ValueError(f"attr {key!r}: {name} is a mapping of class to value")  # noqa: TRY004
        for cls, value in per_class.items():
            if not isinstance(cls, str) or not CLASS_NAME.fullmatch(cls):
                raise ValueError(
                    f"attr {key!r}: {name} has the class {str(cls)[:64]!r}; a class "
                    "name is at most 64 letters, digits, _ - . and spaces"
                )
            number(f"{name}/{cls}", value)


def check_confusion(confusion: Any) -> None:
    """The user attr `confusion`: the names of the classes, and per true class the count
    of test rows predicted as each class, over every repeat of the run."""
    if not isinstance(confusion, dict) or set(confusion) != {"classes", "counts"}:
        raise ValueError("attr 'confusion' is a mapping of 'classes' and 'counts'")
    classes, counts = confusion["classes"], confusion["counts"]
    if not isinstance(classes, list) or not all(
        isinstance(c, str) and CLASS_NAME.fullmatch(c) for c in classes
    ):
        raise ValueError(
            "attr 'confusion': classes is a list of class names, each at most 64 "
            "letters, digits, _ - . and spaces"
        )
    n = len(classes)
    if (
        not isinstance(counts, list)
        or len(counts) != n
        or not all(isinstance(row, list) and len(row) == n for row in counts)
        or not all(is_number(v, integer=True) and v >= 0 for row in counts for v in row)
    ):
        raise ValueError(
            f"attr 'confusion': counts is {n} x {n} whole numbers, one row per class"
        )


def lookup(metrics: dict[str, Any], name: str) -> float | None:
    """One metric out of the attr `metrics` or `metrics_std`: a scalar (`macro_recall`)
    or one class of a metric per class (`recall/F`); None when it is not there."""
    head, _, cls = name.partition("/")
    found: Any = metrics.get(head)
    if cls:
        found = found.get(cls) if isinstance(found, dict) else None
    elif isinstance(found, dict):
        return None
    return float(found) if isinstance(found, int | float) and is_number(found) else None


def is_number(value: Any, integer: bool = False) -> bool:
    """A finite number that sqlite and a plot can hold; with `integer`, a whole one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if integer and not isinstance(value, int):
        return False
    return is_finite(value)


def is_finite(value: Any) -> bool:
    """False for NaN, an infinity, and a number beyond what sqlite stores."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return True
    if isinstance(value, int):
        return abs(value) <= MAX_INT
    return math.isfinite(value) and abs(value) <= MAX_INT


def builds_on(attrs: dict[str, Any]) -> list[str]:
    """The submission_refs the user attr `builds_on` names, comma-separated; [] without
    it."""
    value = attrs.get("builds_on")
    return [] if not isinstance(value, str) else value.split(",")


def shown(path: str) -> str:
    """A path as it is safe to print: quoted when it holds anything unprintable."""
    return path if path.isprintable() else repr(path)


def check_keys(kind: str, keys: Any) -> None:
    for key in keys:
        if not KEY.fullmatch(key):
            raise ValueError(
                f"{kind} key {key!r}: only letters, digits, _ - . / and spaces, "
                "at most 250 characters"
            )


def check_json_size(kind: str, mapping: dict[str, Any]) -> None:
    for key, value in mapping.items():
        size = len(json.dumps(value))
        if size > MAX_ATTR_BYTES:
            raise ValueError(f"{kind} {key!r} is {size} bytes, limit {MAX_ATTR_BYTES}")


class Point(BaseModel):
    """The value a trial reported at one step: the accuracy after one epoch, or of one
    repeat of a run."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    step: int = Field(ge=0, le=MAX_INT)
    value: float


class Distribution(BaseModel):
    """What a param was drawn from, as `optuna.distributions.distribution_to_json`
    writes it: a kind and its attributes."""

    model_config = ConfigDict(extra="forbid")
    name: Literal["FloatDistribution", "IntDistribution", "CategoricalDistribution"]
    attributes: dict[str, Any]

    @model_validator(mode="after")
    def _attributes(self) -> Distribution:
        if set(self.attributes) != DISTRIBUTIONS[self.name]:
            raise ValueError(
                f"{self.name} has the attributes {sorted(DISTRIBUTIONS[self.name])}, "
                f"got {sorted(self.attributes)}"
            )
        if len(json.dumps(self.attributes)) > MAX_ATTR_BYTES:
            raise ValueError(f"{self.name} is larger than {MAX_ATTR_BYTES} bytes")
        a = self.attributes
        if self.name == "CategoricalDistribution":
            choices = a["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError("choices is a list with at least one value")
            if not all(c is None or isinstance(c, Scalar) for c in choices):
                raise ValueError("every choice is text, a number, true, false or null")
            if not all(is_finite(c) for c in choices):
                raise ValueError("a choice is a finite number")
            return self
        integer = self.name == "IntDistribution"
        for end in ("low", "high"):
            if not is_number(a[end], integer):
                kind = "a whole number" if integer else "a finite number"
                raise ValueError(f"{end} is {kind}")
        if a["low"] > a["high"]:
            raise ValueError("low is at most high")
        if not isinstance(a["log"], bool):
            raise ValueError("log is true or false")  # noqa: TRY004
        step = a["step"]
        if integer and not (is_number(step, True) and step >= 1):
            raise ValueError("the step of an IntDistribution is a whole number from 1")
        if not integer and step is not None and not (is_number(step) and step > 0):
            raise ValueError("the step of a FloatDistribution is null or above 0")
        if a["log"] and (a["low"] <= 0 or (step not in (None, 1))):
            raise ValueError("a log range starts above 0 and has no step")
        return self

    def holds(self, value: Any) -> bool:
        """Whether `value` is one this distribution can give, as Optuna decides it."""
        a = self.attributes
        if self.name == "CategoricalDistribution":
            return any(type(c) is type(value) and c == value for c in a["choices"])
        integer = self.name == "IntDistribution"
        if not is_number(value, integer) or not a["low"] <= value <= a["high"]:
            return False
        if a["step"] is None:
            return True
        steps = (value - a["low"]) / a["step"]
        return abs(steps - round(steps)) < 1e-8


class StudyInfo(BaseModel):
    """The study a trial belongs to: its direction and its user attrs (the search space
    of a search, the study's name, its budget, ...)."""

    model_config = ConfigDict(extra="forbid")
    direction: Literal["maximize", "minimize"]
    attrs: dict[str, Any]

    @field_validator("attrs")
    @classmethod
    def _attrs(cls, v: dict[str, Any]) -> dict[str, Any]:
        check_keys("study attr", v)
        check_json_size("study attr", v)
        check_idea(v)
        objective = v.get("objective", DEFAULT_OBJECTIVE)
        if objective not in OBJECTIVES:
            raise ValueError(
                f"study attr 'objective': {str(objective)[:40]!r} is not one of {OBJECTIVES}"
            )
        return v


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run_id: str
    run_name: str
    experiment: str = Field(description="the name of the study the trial belongs to")
    status: Literal["COMPLETE", "PRUNED", "FAIL"]
    start_time: int = Field(
        ge=0, le=MAX_INT, description="milliseconds since the epoch"
    )
    end_time: int | None = Field(ge=0, le=MAX_INT)
    params: dict[str, Scalar]
    distributions: dict[str, Distribution] = Field(
        description="per param: what it was drawn from; absent for a param that was given"
    )
    value: float | None
    intermediate: list[Point]
    attrs: dict[str, Any] = Field(
        description="the trial's user attrs: role, dataset, n_params, architecture, ..."
    )
    study: StudyInfo
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="artifact path -> parsed JSON for .json files, text otherwise",
    )

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, v: str) -> str:
        if not RUN_ID.fullmatch(v):
            raise ValueError("run_id must be a 32-character hex id")
        return v

    @field_validator("experiment")
    @classmethod
    def _experiment(cls, v: str) -> str:
        if not EXPERIMENT.fullmatch(v):
            raise ValueError("experiment must be named lesson<N>-<lowercase-name>")
        return v

    @field_validator("params")
    @classmethod
    def _params(cls, v: dict[str, Scalar]) -> dict[str, Scalar]:
        check_keys("param", v)
        check_json_size("param", v)
        for key, value in v.items():
            if not is_finite(value):
                raise ValueError(f"param {key!r} is a finite number")
        return v

    @field_validator("attrs")
    @classmethod
    def _attrs(cls, v: dict[str, Any]) -> dict[str, Any]:
        check_keys("attr", v)
        check_json_size("attr", v)
        if "metrics" in v:
            check_metrics("metrics", v["metrics"])
        if "metrics_std" in v:
            check_metrics("metrics_std", v["metrics_std"], spread=True)
        if "confusion" in v:
            check_confusion(v["confusion"])
        for ref in builds_on(v):
            if not SUBMISSION_REF.fullmatch(ref):
                raise ValueError(
                    f"attr 'builds_on': {ref!r} is not a submission_ref "
                    "lesson<N>/<login>/<run_id>; copy it from the class run"
                )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> Run:
        if self.status == "COMPLETE" and self.value is None:
            raise ValueError("a COMPLETE trial has a value")
        unknown = sorted(set(self.distributions) - set(self.params))
        if unknown:
            raise ValueError(f"distributions for params that are absent: {unknown}")
        check_keys("distribution", self.distributions)
        for key, distribution in self.distributions.items():
            if not distribution.holds(self.params[key]):
                raise ValueError(
                    f"param {key!r} is not a value of the distribution it names"
                )
        if HANDED_OUT.fullmatch(self.experiment) and self.study.direction != "maximize":
            raise ValueError("a handed-out study maximizes")
        objective = self.study.attrs.get("objective", DEFAULT_OBJECTIVE)
        logged = self.attrs.get("metrics", {}).get(objective)
        if (
            self.status == "COMPLETE"
            and self.value is not None
            and logged is not None
            and abs(self.value - logged) > VALUE_TOLERANCE
        ):
            raise ValueError(
                f"the value {self.value:.5g} is not the objective {objective} "
                f"{logged:.5g} the attr 'metrics' logs"
            )
        return self

    @field_validator("artifacts")
    @classmethod
    def _artifacts(cls, v: dict[str, Any]) -> dict[str, Any]:
        for path, content in v.items():
            parts = path.split("/")
            if not ARTIFACT_PATH.fullmatch(path) or any(
                p in (".", "..") for p in parts
            ):
                raise ValueError(
                    f"artifact {path[:80]!r}: the path must be relative, without . or "
                    "..; only letters, digits, _ - . and / in it"
                )
            if not path.endswith(ARTIFACT_SUFFIXES):
                raise ValueError(
                    f"artifact {path!r}: only {ARTIFACT_SUFFIXES} files are submitted"
                )
            if not path.endswith(".json") and not isinstance(content, str):
                raise ValueError(f"artifact {path!r}: a file that is not .json is text")
            size = len(json.dumps(content))
            if size > MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"artifact {path!r} is {size} bytes, limit {MAX_ARTIFACT_BYTES}"
                )
        return v


class Code(BaseModel):
    """Where the code came from: the course checkout's HEAD when the run was submitted."""

    model_config = ConfigDict(extra="forbid")
    commit: str | None
    dirty: bool | None
    remote: str | None


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_: Literal["mads-run/2"] = Field(alias="schema")
    lesson: int = Field(ge=1, le=8)
    student: str
    submitted_at: datetime
    code: Code
    run: Run

    @field_validator("student")
    @classmethod
    def _student(cls, v: str) -> str:
        if not LOGIN.fullmatch(v):
            raise ValueError("student must be a GitHub login")
        return v

    @model_validator(mode="after")
    def _not_on_itself(self) -> Submission:
        if self.ref().lower() in (r.lower() for r in builds_on(self.run.attrs)):
            raise ValueError("attr 'builds_on' names this run itself")
        return self

    @model_validator(mode="after")
    def _own_idea(self) -> Submission:
        author = self.run.study.attrs.get("idea_author")
        if author is None or author == self.student.lower():
            return self
        if author == COURSE and HANDED_OUT.fullmatch(self.run.experiment):
            return self
        raise ValueError(
            f"study attr 'idea_author' is {author!r}: an idea is its submitter's "
            f"({self.student.lower()!r}), or {COURSE!r} for a study a lesson hands out"
        )

    def idea(self) -> tuple[str, str, list[str]] | None:
        """(name, id, parents' names) of the idea this run's study carries, if any."""
        attrs = self.run.study.attrs
        if "idea_name" not in attrs:
            return None
        return attrs["idea_name"], attrs["idea_id"], idea_parents(attrs["idea_parents"])

    def objective(self) -> str:
        """The metric that is this run's value: its study's objective."""
        return self.run.study.attrs.get("objective", DEFAULT_OBJECTIVE)

    def metric(self, name: str) -> float | None:
        """One metric of this run: a scalar (`macro_recall`) or one class of a metric per
        class (`recall/F`). Read from the attr `metrics`; for a run without it, the
        value stands for its objective. None when the run does not log it."""
        logged = self.run.attrs.get("metrics")
        if isinstance(logged, dict):
            return lookup(logged, name)
        if name == self.objective() and self.run.value is not None:
            return float(self.run.value)
        return None

    def spread(self, name: str) -> float | None:
        """The spread over the repeats of one metric, as `metric` names it; None for a
        run that was not repeated or does not log it."""
        logged = self.run.attrs.get("metrics_std")
        if isinstance(logged, dict):
            return lookup(logged, name)
        found = self.run.attrs.get(f"{name}_std")
        if "/" in name or not isinstance(found, int | float) or not is_number(found):
            return None
        return float(found)

    def metric_names(self) -> list[str]:
        """Every metric this run logs, as `metric` names them."""
        logged = self.run.attrs.get("metrics")
        if not isinstance(logged, dict):
            return [self.objective()] if self.run.value is not None else []
        names = [m for m in SCALAR_METRICS if m in logged]
        for m in PER_CLASS_METRICS:
            if isinstance(logged.get(m), dict):
                names += [f"{m}/{cls}" for cls in logged[m]]
        return names

    def ref(self) -> str:
        """The submission_ref: where the file sits, without `.json`."""
        return f"lesson{self.lesson}/{self.student}/{self.run.run_id}"

    def relpath(self) -> str:
        return f"{self.ref()}.json"

    def dump(self) -> str:
        return (
            json.dumps(
                self.model_dump(mode="json", by_alias=True), indent=1, sort_keys=True
            )
            + "\n"
        )

    def content_hash(self) -> str:
        """Hash of the run itself, independent of when it was submitted."""
        body = json.dumps(self.run.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def copy_hash(self) -> str:
        """Hash of the run without its id, its name and the student: two files with the
        same copy hash hold the same run, however they are labelled."""
        run = self.run.model_dump(mode="json", exclude={"run_id", "run_name"})
        body = json.dumps(run, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def load(path: Path, root: Path | None = None) -> Submission:
    """Read and validate one submission file. With `root`, also check that the file sits
    at the path its own content says it belongs at."""
    if path.is_symlink():
        raise ValueError(
            "a symbolic link, not a file: submit again, do not edit by hand"
        )
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"{size} bytes, limit {MAX_FILE_BYTES}")
    rel = None
    if root is not None:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    return parse(path.read_bytes(), rel)


TOO_DEEP = "nested too deep: submit again, do not edit by hand"


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """A JSON object as a dict; a key that is there twice is an error, since readers
    would disagree on which one counts."""
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("a key is there twice: submit again, do not edit by hand")
    return out


def parse(raw: bytes, rel: str | None = None) -> Submission:
    """Validate the bytes of one submission file. With `rel`, the path of the file in the
    runs repository, also check that it sits where its own content says it belongs."""
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"{len(raw)} bytes, limit {MAX_FILE_BYTES}")
    try:
        data = json.loads(raw.decode(), object_pairs_hook=unique_keys)
        # a lone surrogate decodes, and cannot be written again
        json.dumps(data, ensure_ascii=False).encode()
    except (json.JSONDecodeError, UnicodeError) as e:
        raise ValueError(
            f"not valid JSON ({e}): submit again, do not edit by hand"
        ) from None
    except RecursionError:
        raise ValueError(TOO_DEEP) from None
    sub = Submission.model_validate(data)
    try:
        sub.copy_hash()
    except ValueError:  # pydantic refuses to serialize beyond its depth
        raise ValueError(TOO_DEEP) from None
    if rel is not None:
        m = PATH.fullmatch(rel)
        if m and m.group(2) != sub.student:
            raise ValueError(
                f"this file says it is {sub.student}'s run; "
                f"it sits in {m.group(2)}'s directory"
            )
        if rel != sub.relpath():
            raise ValueError(f"belongs at {sub.relpath()}, found at {rel}")
    return sub


class Index:
    """What a check of one file needs from all the others: per copy hash the students
    whose directories hold a file with it, and per idea name the ids it stands for."""

    def __init__(self, root: Path) -> None:
        self.copies: dict[str, set[str]] = {}
        self.ideas: dict[str, set[str]] = {}
        for path in root.glob("lesson*/*/*.json"):
            try:
                sub = load(path)
            except ValueError:
                continue
            self.copies.setdefault(sub.copy_hash(), set()).add(path.parent.name)
            idea = sub.idea()
            if idea:
                self.ideas.setdefault(idea[0], set()).add(idea[1])


def check(
    root: Path, files: list[str] | None, actor: str | None, teachers: set[str]
) -> list[str]:
    """Every problem with the submissions under `root`, as readable lines.

    `files` restricts the check to those paths (the files a push or pull request
    changed), and then every one of them must be a `lesson<N>/<student>/<run_id>.json`
    path, checked before anything else; with them, `actor` adds the rule that a student
    only writes inside the directory spelled exactly as their login. A run, by id or by
    content, may sit in one student's directory only. An idea builds on ideas that are
    under `root`, and its name stands for one id there."""
    problems = []
    if files is None:
        rels = sorted(
            p.relative_to(root).as_posix() for p in root.glob("lesson*/*/*.json")
        )
    else:
        rels = [f for f in files if f]
    index: Index | None = None

    for rel in rels:
        m = PATH.fullmatch(rel)
        if not m:
            problems.append(
                f"{shown(rel)}: outside lesson<N>/<student>/<run_id>.json; "
                "a submission changes nothing else"
            )
            continue
        path = root / rel
        owner, run_id = m.group(2), m.group(3)
        deleted = not path.exists() and not path.is_symlink()
        if (
            files is not None
            and actor
            and actor.lower() not in teachers
            and owner != actor
        ):
            verb = "deleted" if deleted else "written"
            problems.append(
                f"{rel}: {verb} by {actor}, but this directory belongs to {owner}"
            )
        if deleted:
            continue
        others = sorted(
            {
                p.parent.name
                for p in root.glob(f"lesson*/*/{run_id}.json")
                if p.parent.name != owner
            }
        )
        if others:
            problems.append(
                f"{rel}: this run id is already submitted by {', '.join(others)}"
            )
        try:
            sub = load(path, root)
        except ValueError as e:  # pydantic's ValidationError is a ValueError
            problems.extend(f"{rel}: {line}" for line in explain(e))
            continue
        if index is None:
            index = Index(root)
        copied = sorted(index.copies.get(sub.copy_hash(), set()) - {owner, *others})
        if copied:
            problems.append(
                f"{rel}: this run is already submitted by {', '.join(copied)}, "
                "under another run id"
            )
        idea = sub.idea()
        if idea is None:
            continue
        name, _, parents = idea
        if len(index.ideas.get(name, ())) > 1:
            problems.append(
                f"{rel}: the name {name} already stands for another idea in the "
                "class repository; change a word of your idea and run the study again"
            )
        for parent in parents:
            if parent not in index.ideas:
                problems.append(
                    f"{rel}: builds on the idea {parent}, which is not in the class "
                    "repository; an idea builds on ideas that are published"
                )
    return problems


def read_files(files: list[str] | None, files_from: Path | None) -> list[str] | None:
    """The changed paths: from a file of NUL-separated paths, or from `--files`
    arguments split on newlines only. None when neither is given."""
    if files_from is not None:
        raw = files_from.read_bytes().decode("utf-8", errors="surrogateescape")
        return [f for f in raw.split("\0") if f]
    if files is None:
        return None
    return [f for arg in files for f in arg.split("\n") if f]


def explain(error: Exception) -> list[str]:
    """One readable line per problem in a failed `load`."""
    if isinstance(error, ValidationError):
        return [
            f"{'.'.join(shown(str(p)) for p in err['loc'])}: {err['msg']}"
            if err["loc"]
            else err["msg"]
            for err in error.errors()
        ]
    return [str(error)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate the submissions in a runs repository")
    c.add_argument("root", type=Path)
    c.add_argument(
        "--files", nargs="*", help="only these paths, relative to root (one per line)"
    )
    c.add_argument(
        "--files-from",
        type=Path,
        help="only the paths in this file, NUL-separated, relative to root",
    )
    c.add_argument(
        "--actor",
        help="GitHub login that pushed or opened the pull request; "
        "may only write in its own directory",
    )
    c.add_argument(
        "--teachers", default="", help="comma-separated logins that may write anywhere"
    )
    sub.add_parser("schema", help="print the JSON schema of one submission file")
    args = parser.parse_args()

    if args.cmd == "schema":
        print(json.dumps(Submission.model_json_schema(by_alias=True), indent=2))
        return

    teachers = {t.strip().lower() for t in args.teachers.split(",") if t.strip()}
    files = read_files(args.files, args.files_from)
    if files is not None and len(files) > MAX_CHANGED_FILES:
        too_many = (
            f"the change touches {len(files)} files, more than {MAX_CHANGED_FILES}"
        )
        problems = [f"{too_many}; submit in smaller parts"]
    else:
        problems = check(args.root, files, args.actor, teachers)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
