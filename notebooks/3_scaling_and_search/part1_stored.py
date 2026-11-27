"""The stored run of Part 1 of `05_exercise.ipynb`, for a student who runs it with
`QUICK = True`: `part1_stored.json`, the rows of one full run of Part 1.

    uv run notebooks/3_scaling_and_search/part1_stored.py

It reads the rows the notebook stored in `artefacts/lesson3/part1.json` and keeps E1 to E3
and the trials of the search as it stands in `lesson3-notebook`, so run Part 1 in full
first. Every kept row says its cap (`per_class`) and its train rows (`n_train`); a row
without them is left out and named. Nothing is trained here.
"""

import json
from pathlib import Path

from optuna.trial import TrialState

from hpo.record import Store, StudyLog

# what exercises/harness.py calls NOTEBOOK, and the file its results() writes
NOTEBOOK = "lesson3-notebook"
RESULTS = Store.default().root.parent / "lesson3" / "part1.json"

OUT = Path(__file__).parent / "part1_stored.json"


def main() -> None:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))
    trials = {
        f"trial {t.number}" for t in StudyLog.load(NOTEBOOK).trials(TrialState.COMPLETE)
    }
    capped = [r for r in rows.values() if {"per_class", "n_train"} <= r.keys()]
    capless = [r["name"] for r in rows.values() if r not in capped]
    if capless:
        print(f"left out, without per_class and n_train: {', '.join(capless)}")
    kept = [r for r in capped if r["experiment"] != "E4" or r["name"] in trials]
    OUT.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
    counts = {
        e: sum(r["experiment"] == e for r in kept) for e in ("E1", "E2", "E3", "E4")
    }
    print(f"written: {OUT.name}, {counts}")


if __name__ == "__main__":
    main()
