"""The names hpo shares with the class run protocol.

`scripts/run_submission.py` is the protocol's one source: it is copied into the class
repository, where it validates a submitted run with pydantic and the standard library
alone. hpo reads its recipe for an idea and its limits from it, so the two cannot drift.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from run_submission import (  # ty: ignore[import-not-found]  (scripts/ is put on sys.path above)
    COURSE,
    IDEA_ATTRS,
    IDEA_NAME,
    LOGIN,
    MAX_ARTIFACT_BYTES,
    MAX_IDEA_CHARS,
    MIN_IDEA_CHARS,
    SUBMISSION_REF,
    idea_id,
    idea_name,
    idea_words,
)

__all__ = [
    "COURSE",
    "IDEA_ATTRS",
    "IDEA_NAME",
    "LOGIN",
    "MAX_ARTIFACT_BYTES",
    "MAX_IDEA_CHARS",
    "MIN_IDEA_CHARS",
    "SUBMISSION_REF",
    "idea_id",
    "idea_name",
    "idea_words",
]
