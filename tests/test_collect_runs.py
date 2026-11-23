"""collect_runs against the git world of `gitworld.py`: it reads the class repository's
main from a git ref, whatever branch the clone is on."""

import sys
from pathlib import Path

import collect_runs  # ty: ignore[import-not-found]
import optuna
import submit_runs  # ty: ignore[import-not-found]
from gitworld import run  # ty: ignore[import-not-found]  (conftest puts scripts/ on sys.path)
from optuna.artifacts import FileSystemArtifactStore, get_all_artifact_meta
from optuna.distributions import FloatDistribution


def collect(monkeypatch, repo: Path, store: Path, *extra: str) -> None:
    argv = ["collect_runs.py", "--runs-repo", str(repo), "--store", str(store)]
    monkeypatch.setattr(sys, "argv", [*argv, *extra])
    collect_runs.main()


def trials(store: Path) -> dict[str, optuna.trial.FrozenTrial]:
    """submission_ref -> trial, over every study of the class store."""
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    out = {}
    for summary in optuna.get_all_study_summaries(f"sqlite:///{store}"):
        study = optuna.load_study(
            study_name=summary.study_name, storage=f"sqlite:///{store}"
        )
        for trial in study.trials:
            out[trial.user_attrs["submission_ref"]] = trial
    return out


def studies(store: Path) -> list[str]:
    summaries = optuna.get_all_study_summaries(f"sqlite:///{store}")
    return sorted(s.study_name for s in summaries)


def imported(store: Path) -> dict[str, dict]:
    """submission_ref -> user attrs of every trial in the class store."""
    return {ref: t.user_attrs for ref, t in trials(store).items()}


def ref(sub) -> str:
    return sub.relpath().removesuffix(".json")


def state(repo: Path) -> tuple[str, str, str]:
    """The branch, the commit and the working tree's status of a clone."""
    return (
        run("git", "branch", "--show-current", cwd=repo),
        run("git", "rev-parse", "HEAD", cwd=repo),
        run("git", "status", "--porcelain", "--untracked-files=all", cwd=repo),
    )


def teacher_clone(world) -> Path:
    """The teacher's clone of the class repository: only `origin`."""
    where = world.tmp / "class-clone"
    run("git", "clone", "--quiet", str(world.upstream), str(where))
    return where


def test_a_students_clone_on_its_lesson_branch_reads_upstream_main(
    world, submission, tmp_path, monkeypatch
):
    merged, pending = submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [merged])
    world.squash_merge()
    bob = submission("bob")
    world.teacher_push(bob)
    submit_runs.submit(world.clone, 1, "alice", [merged, pending])
    carol = submission("carol")
    world.teacher_push(carol)  # after the clone last fetched: collect must fetch it
    (world.clone / "notes.txt").write_text("mine\n", encoding="utf-8")
    before = state(world.clone)
    assert before[0] == "lesson1"

    uri = tmp_path / "class.db"
    collect(monkeypatch, world.clone, uri, "--lesson", "1")
    # the fork's main has no runs and lesson1 holds the pending one: neither is read
    assert sorted(imported(uri)) == sorted([ref(merged), ref(bob), ref(carol)])
    assert imported(uri)[ref(bob)]["student"] == "bob"
    assert state(world.clone) == before


def test_the_teachers_clone_with_only_origin(world, submission, tmp_path, monkeypatch):
    first = submission("alice")
    world.teacher_push(first)
    clone = teacher_clone(world)
    run("git", "switch", "--quiet", "-c", "notes", cwd=clone)
    later = submission("bob")
    world.teacher_push(later)
    before = state(clone)

    uri = tmp_path / "class.db"
    collect(monkeypatch, clone, uri, "--no-pull")
    assert sorted(imported(uri)) == [ref(first)]
    collect(monkeypatch, clone, uri)
    assert sorted(imported(uri)) == sorted([ref(first), ref(later)])
    assert state(clone) == before
    assert not (clone / later.relpath()).exists()


def test_the_design_script_lands_as_design_py(world, submission, tmp_path, monkeypatch):
    sub = submission("alice")
    design = "import torch\n\nmodel = torch.nn.Linear(4, 2)\n"
    run_ = sub.run.model_copy(update={"artifacts": {"design.py": design}})
    sub = sub.model_copy(update={"run": run_})
    world.teacher_push(sub)

    store = tmp_path / "class.db"
    collect(monkeypatch, teacher_clone(world), store)
    [only] = trials(store).values()
    assert only.user_attrs["submission_ref"] == ref(sub)
    storage = optuna.storages.RDBStorage(f"sqlite:///{store}")
    [meta] = get_all_artifact_meta(only, storage=storage)
    assert meta.filename == "design.py"
    reader = FileSystemArtifactStore(str(tmp_path / "class-artifacts")).open_reader(
        meta.artifact_id
    )
    assert reader.read().decode() == design


def test_the_architecture_becomes_the_trials_note(
    world, submission, tmp_path, monkeypatch
):
    sub = submission("alice")
    arch = {"signature": "Linear(4->2)", "summary": "layer table"}
    run_ = sub.run.model_copy(update={"artifacts": {"architecture.json": arch}})
    world.teacher_push(sub.model_copy(update={"run": run_}))
    store = tmp_path / "class.db"
    collect(monkeypatch, teacher_clone(world), store)

    from optuna_dashboard._note import get_note_from_system_attrs

    storage = optuna.storages.RDBStorage(f"sqlite:///{store}")
    (study_id,) = [storage.get_study_id_from_name(n) for n in studies(store)]
    attrs = storage.get_study_system_attrs(study_id)
    [trial] = trials(store).values()
    note = get_note_from_system_attrs(attrs, trial._trial_id)
    assert "Linear(4->2)" in note["body"] and "layer table" in note["body"]


def test_the_handed_out_studies_are_pooled_and_the_others_kept_per_student(
    world, submission, tmp_path, monkeypatch
):
    def in_study(sub, name):
        return sub.model_copy(
            update={"run": sub.run.model_copy(update={"experiment": name})}
        )

    for sub in (
        in_study(submission("alice"), "lesson1-notebook"),
        in_study(submission("bob"), "lesson1-notebook"),
        in_study(submission("alice"), "lesson1-notebook-runs"),
        in_study(submission("alice"), "lesson1-study-alice-depth"),
        in_study(submission("bob"), "lesson1-study-bob-width"),
    ):
        world.teacher_push(sub)
    store = tmp_path / "class.db"
    collect(monkeypatch, teacher_clone(world), store)
    assert studies(store) == [
        "lesson1-notebook",
        "lesson1-notebook-runs",
        "lesson1-study-alice-depth@alice",
        "lesson1-study-bob-width@bob",
    ]
    pooled = {
        t.user_attrs["student"]
        for t in trials(store).values()
        if t.user_attrs["lesson"] == 1
    }
    assert pooled == {"alice", "bob"}


def test_a_trial_that_does_not_fit_the_pooled_study_goes_to_the_students_own(
    world, submission, tmp_path, monkeypatch
):
    ok, odd = submission("alice"), submission("bob")
    run_ = odd.run.model_copy(
        update={"params": {"lr": "fast", "encoder": "dinov2-small"}}
    )
    world.teacher_push(ok)
    world.teacher_push(odd.model_copy(update={"run": run_}))
    store = tmp_path / "class.db"
    collect(monkeypatch, teacher_clone(world), store)
    assert studies(store) == ["lesson1-notebook-runs", "lesson1-notebook-runs@bob"]


def test_collecting_again_rebuilds_exactly_what_main_holds(
    world, submission, tmp_path, monkeypatch, capsys
):
    first = submission("alice")
    world.teacher_push(first)
    clone = teacher_clone(world)
    store = tmp_path / "class.db"
    collect(monkeypatch, clone, store)
    collect(monkeypatch, clone, store)
    assert sorted(imported(store)) == [ref(first)]
    assert "1 new" in capsys.readouterr().out


def test_a_trial_keeps_what_the_run_says(world, submission, tmp_path, monkeypatch):
    sub = submission("alice")
    run_ = sub.run.model_copy(
        update={
            "status": "PRUNED",
            "value": 0.4,
            "params": {"lr": 0.003},
            "distributions": {
                "lr": {
                    "name": "FloatDistribution",
                    "attributes": {
                        "low": 0.0001,
                        "high": 0.01,
                        "log": True,
                        "step": None,
                    },
                }
            },
            "intermediate": [
                {"step": 0, "value": 0.3},
                {"step": 1, "value": 0.4},
            ],
        }
    )
    world.teacher_push(sub.model_copy(update={"run": run_}))
    store = tmp_path / "class.db"
    collect(monkeypatch, teacher_clone(world), store)
    [t] = trials(store).values()
    assert t.state == optuna.trial.TrialState.PRUNED
    assert t.params == {"lr": 0.003}
    assert t.intermediate_values == {0: 0.3, 1: 0.4}
    assert t.distributions["lr"] == FloatDistribution(0.0001, 0.01, log=True)


def test_only_the_lessons_asked_for(world, submission, tmp_path, monkeypatch):
    one, two = submission("alice", lesson=1), submission("bob", lesson=2)
    world.teacher_push(one)
    world.teacher_push(two)
    store = tmp_path / "class.db"
    collect(monkeypatch, teacher_clone(world), store, "--lesson", "1")
    assert sorted(imported(store)) == [ref(one)]


def test_submit_after_collect_on_a_fork_clone(world, submission, tmp_path, monkeypatch):
    first, second = submission("alice"), submission("alice")
    submit_runs.submit(world.clone, 1, "alice", [first])
    world.squash_merge()
    world.teacher_push(submission("bob"))
    collect(monkeypatch, world.clone, tmp_path / "class.db")

    submit_runs.submit(world.clone, 1, "alice", [first, second])
    assert second.relpath() in world.files_in_pr()
    assert run("git", "branch", "--show-current", cwd=world.clone) == "lesson1"
