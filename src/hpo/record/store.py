"""Where the course keeps what a student runs: the one module that decides the paths.

The studies live in one sqlite file, the artifacts (the layer table, the design script)
in a folder beside it, and the student's GitHub login in a small file there too. A run
under notebooktester gets a store of its own (`test.db`, `test-artifacts`), so a test
never mixes with a search.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hpo.record.protocol import LOGIN

# the environment variable notebooktester sets while it runs a notebook or a script
TESTER = "NOTEBOOKTESTER_RUNNING"
# the store of the course repository this package is installed from
COURSE_STORE = Path(__file__).resolve().parents[3] / "artefacts" / "optuna"


def under_tester() -> bool:
    """True while notebooktester runs: placeholders pass, and the store is the tester's."""
    return bool(os.environ.get(TESTER))


@dataclass(frozen=True)
class Store:
    """The sqlite file of the studies, the artifact folder and the kept login, under
    `root`."""

    root: Path
    testing: bool = False

    @classmethod
    def default(cls) -> Store:
        """`artefacts/optuna` at the root of the course repository, the tester's store
        while notebooktester runs."""
        return cls(COURSE_STORE, testing=under_tester())

    @property
    def db(self) -> Path:
        return self.root / ("test.db" if self.testing else "studies.db")

    @property
    def url(self) -> str:
        """The sqlite URL Optuna opens; the folder is made on first use."""
        self.root.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.db}"

    @property
    def artifacts(self) -> Path:
        return self.root / ("test-artifacts" if self.testing else "artifacts")

    def logdir(self) -> Path:
        """Where a trainer may write its checkpoints; never made unless one is saved."""
        return self.root.parent / "modellogs"

    def login(self) -> str:
        """The student's GitHub login in lowercase, the author of their ideas: read once
        through `gh` and kept in `login`. Under notebooktester it is `tester`."""
        if self.testing:
            return "tester"
        kept = self.root / "login"
        if kept.exists() and LOGIN.fullmatch(kept.read_text(encoding="utf-8").strip()):
            return kept.read_text(encoding="utf-8").strip()
        try:
            found = subprocess.run(
                ["gh", "api", "user", "--jq", ".login"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                encoding="utf-8",
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            found = ""
        if not LOGIN.fullmatch(found):
            raise ValueError(
                "idea: gh could not say who you are (is it installed and logged in? "
                "`gh auth login`); or pass your GitHub login: Idea.of(..., author='<login>')"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        kept.write_text(found.lower() + "\n", encoding="utf-8")
        return found.lower()
