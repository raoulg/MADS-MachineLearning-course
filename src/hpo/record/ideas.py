"""What a study is called, what it investigates, and which ideas of the class it builds on.

An idea's id is a hash of its author, its words and the names of its parents; its name
(`blissful-bee-18`) is that id made readable, and is what a classmate passes as
`builds_on`. The recipe is the class run protocol's (`hpo.record.protocol`), so a submitted
study can be checked without hpo.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from hpo.record.protocol import (
    COURSE,
    IDEA_ATTRS,
    IDEA_NAME,
    LOGIN,
    MAX_IDEA_CHARS,
    MIN_IDEA_CHARS,
    idea_id,
    idea_name,
    idea_words,
)
from hpo.record.store import Store, under_tester

STUDY_PLACEHOLDER = "name-your-study"
IDEA_PLACEHOLDER = "say what your study investigates"
# <first name>-<what it studies>: who it is from, and what it is about
STUDY_NAME = re.compile(r"[a-z]+-[a-z0-9][a-z0-9-]{1,38}")


def study_name(name: str) -> str:
    """`name`, stripped and in lowercase, if it can name a study; otherwise a
    `ValueError` that says what to do. The placeholder passes only under notebooktester."""
    name = name.strip().lower()
    if not name or (name == STUDY_PLACEHOLDER and not under_tester()):
        raise ValueError(
            f"name your study: set STUDY to your first name and what it studies "
            f"instead of {STUDY_PLACEHOLDER!r}, for example 'sam-depth-on-eurosat'"
        )
    if not STUDY_NAME.fullmatch(name):
        raise ValueError(
            f"study name {name!r}: use your first name, a dash, and what the study "
            "studies, in lowercase letters, digits and dashes, for example "
            "'sam-depth-on-eurosat'"
        )
    return name


def _parents(builds_on: str | Sequence[str] | None) -> list[str]:
    names = [builds_on] if isinstance(builds_on, str) else list(builds_on or [])
    parents = sorted(
        {n.strip() for part in names for n in part.split(",") if n.strip()}
    )
    for parent in parents:
        if not IDEA_NAME.fullmatch(parent):
            raise ValueError(
                f"idea: builds_on {parent!r} is not the name of an idea; copy the "
                "name from the class view, like 'blissful-bee-18'"
            )
    return parents


class Idea(BaseModel):
    """The idea of a study, as the study attrs `attrs` gives carry it."""

    model_config = ConfigDict(frozen=True)

    words: str
    author: str
    parents: tuple[str, ...]
    id: str
    name: str

    @classmethod
    def of(
        cls,
        words: str,
        builds_on: str | Sequence[str] | None = None,
        author: str | None = None,
        store: Store | None = None,
    ) -> Idea:
        """The idea `words` of `author`, built on the ideas named in `builds_on`.

        - `words`: what the study investigates, in a few words (`"does depth still help
          on eurosat"`). Other words are another idea.
        - `builds_on`: the name, or the names, of the ideas this one builds on: a
          classmate's from the class view, or an earlier one of your own.
        - `author`: your GitHub login; the store reads it through `gh` when left out.
          `COURSE` for a study a lesson hands out to everyone.

        The placeholder `IDEA_PLACEHOLDER` passes only under notebooktester."""
        said = idea_words(words)
        if said == IDEA_PLACEHOLDER and not under_tester():
            raise ValueError(
                f"idea: say in your own words what your study investigates, instead "
                f"of {IDEA_PLACEHOLDER!r}, for example 'does depth still help on eurosat'"
            )
        if not MIN_IDEA_CHARS <= len(said) <= MAX_IDEA_CHARS:
            raise ValueError(
                f"idea: say what the study investigates in {MIN_IDEA_CHARS} to "
                f"{MAX_IDEA_CHARS} characters, got {said!r}"
            )
        parents = _parents(builds_on)
        who = (author or (store or Store.default()).login()).strip().lower()
        if not LOGIN.fullmatch(who):
            raise ValueError(f"idea: author {who!r} is not a GitHub login")
        id_ = idea_id(who, said, parents)
        return cls(
            words=said, author=who, parents=tuple(parents), id=id_, name=idea_name(id_)
        )

    @classmethod
    def handed_out(cls, lesson: int) -> Idea:
        """The idea of the studies lesson `lesson` hands out to everyone: the root every
        student's first idea of that lesson builds on. Its author is `COURSE`, so it is
        one idea for the whole class."""
        return cls.of(f"lesson {lesson} as handed out", author=COURSE)

    def attrs(self) -> dict[str, str]:
        """The study attrs that carry the idea, under the run protocol's names."""
        values = (self.words, self.author, ",".join(self.parents), self.id, self.name)
        return dict(zip(IDEA_ATTRS, values, strict=True))

    def __str__(self) -> str:
        return f"{self.name}  ({self.words})"
