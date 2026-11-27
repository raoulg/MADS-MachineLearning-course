"""The questions at the top of a lesson's `study.py`, asked at the terminal before the
study runs.

`make study` runs this first:

    uv run python -m hpo.exercise.ask notebooks/<N>_*/exercises/study.py

An answer the study would refuse (a placeholder, a name that is not `<first name>-<what
it studies>`, an idea that is too short) is asked for here and written back into
`study.py`, so the file stays the one place the answers live. Then it shows what will run
and asks whether to start. Without a terminal (a pipe, notebooktester) it asks nothing
and the study's own checks say what is missing.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hpo.exercise.study import EXPECTS_PLACEHOLDER
from hpo.record.ideas import IDEA_PLACEHOLDER, STUDY_PLACEHOLDER, study_name
from hpo.record.protocol import MAX_IDEA_CHARS, MIN_IDEA_CHARS, idea_words
from hpo.record.store import under_tester


@dataclass(frozen=True)
class Literal:
    """A string or number literal in the script: its value and where it stands."""

    value: str | float
    line: int  # 1-based
    start: int  # column offsets in that line
    end: int


def literals(source: str) -> dict[str, Literal]:
    """The literal answers of `source`: `STUDY`, `EXPECTS` and `MINUTES` as assigned,
    and `IDEA` as the first argument of its `Idea.of(...)`."""
    found: dict[str, Literal] = {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if target.id == "IDEA" and isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if (
            target.id in {"STUDY", "IDEA", "EXPECTS", "MINUTES"}
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str | int | float)
            and value.lineno == value.end_lineno
            and value.end_col_offset is not None
        ):
            found[target.id] = Literal(
                value.value, value.lineno, value.col_offset, value.end_col_offset
            )
    return found


def rewrite(source: str, answers: dict[str, tuple[Literal, str]]) -> str:
    """`source` with each literal replaced by its new string."""
    lines = source.splitlines(keepends=True)
    for literal, new in answers.values():
        # the columns ast gives are in UTF-8 bytes
        raw = lines[literal.line - 1].encode()
        lines[literal.line - 1] = (
            raw[: literal.start]
            + json.dumps(new, ensure_ascii=False).encode()
            + raw[literal.end :]
        ).decode()
    return "".join(lines)


def idea_problem(words: str) -> str | None:
    said = idea_words(words)
    if said == IDEA_PLACEHOLDER:
        return "IDEA still holds the placeholder"
    if not MIN_IDEA_CHARS <= len(said) <= MAX_IDEA_CHARS:
        return f"IDEA {said!r} is not {MIN_IDEA_CHARS} to {MAX_IDEA_CHARS} characters"
    return None


def expects_problem(expects: str) -> str | None:
    said = expects.strip()
    if not said or said == EXPECTS_PLACEHOLDER:
        return "EXPECTS still holds the placeholder"
    return None


def study_problem(name: str) -> str | None:
    if name.strip().lower() == STUDY_PLACEHOLDER:
        return "STUDY still holds the placeholder"
    try:
        study_name(name)
    except ValueError as error:
        return str(error)
    return None


def slug(words: str) -> str:
    """`words` as the lowercase, dashed tail of a study name."""
    tail = re.sub(r"[^a-z0-9]+", "-", words.lower()).strip("-")
    return tail[:39].rstrip("-")


def ask(question: str, problem: Callable[[str], str | None]) -> str:
    """Ask `question` until the answer has no problem."""
    while True:
        answer = input(f"{question}\n> ").strip()
        trouble = problem(answer)
        if trouble is None:
            return answer
        print(f"  {trouble}; try again.")


def yes(question: str) -> bool:
    return input(f"{question} [Y/n] ").strip().lower() in {"", "y", "yes", "j", "ja"}


def first_name(answer: str) -> str | None:
    if re.fullmatch(r"[a-z]+", answer.lower()):
        return None
    return "your first name in letters only, like 'sam'"


def questions(script: Path) -> int:
    """Ask what `script` is missing, write the answers back, and ask whether to start.
    The exit code: 0 to run the study, 1 when the student says no."""
    source = script.read_text()
    found = literals(source)
    missing = [key for key in ("STUDY", "IDEA", "EXPECTS") if key not in found]
    if missing:
        # a script that does not keep its answers as plain literals: its own checks
        # say what is wrong
        return 0
    answers: dict[str, tuple[Literal, str]] = {}

    idea = str(found["IDEA"].value)
    if (trouble := idea_problem(idea)) is not None:
        print(f"\n{trouble}.")
        idea = ask(
            "What does your study investigate, in a few words of your own? "
            "For example: does depth still help on eurosat",
            idea_problem,
        )
        answers["IDEA"] = (found["IDEA"], idea)

    expects = str(found["EXPECTS"].value)
    if (trouble := expects_problem(expects)) is not None:
        print(f"\n{trouble}.")
        expects = ask(
            "What do you expect to happen, and why? One line, before you run.",
            expects_problem,
        )
        answers["EXPECTS"] = (found["EXPECTS"], expects)

    study = str(found["STUDY"].value)
    if (trouble := study_problem(study)) is not None:
        print(f"\n{trouble}.")
        first = ask("What is your first name?", first_name).lower()
        study = f"{first}-{slug(idea)}"
        if study_problem(study) is not None or not yes(f"Call the study {study!r}?"):
            study = ask(
                "Name it yourself: your first name, a dash, and what it studies, "
                "like 'sam-depth-on-eurosat'.",
                study_problem,
            )
        answers["STUDY"] = (found["STUDY"], study.strip().lower())

    if answers:
        script.write_text(rewrite(source, answers))
        print(f"\nYour answers are written into {script}.")

    minutes = found["MINUTES"].value if "MINUTES" in found else "?"
    print(
        f"\nstudy    : {study.strip().lower()}\n"
        f"idea     : {idea_words(idea)}\n"
        f"expects  : {expects.strip()}\n"
        f"minutes  : {minutes}  (the search stops when they are up; "
        "run make study again to go on)\n"
    )
    if yes("Start the study?"):
        return 0
    print(f"Not started. Change {script} and run make study again.")
    return 1


def main() -> int:
    script = Path(sys.argv[1])
    if under_tester() or not sys.stdin.isatty():
        return 0
    try:
        return questions(script)
    except (EOFError, KeyboardInterrupt):
        print("\nNot started.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
