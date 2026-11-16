"""optunalog against a temporary sqlite store."""

import json
import sys
from datetime import timedelta
from pathlib import Path

import optuna
import pytest
import torch
from torch import nn

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "notebooks/1_pytorch_intro")
)

import optunalog  # ty: ignore[import-not-found]  (the lesson folder is put on sys.path above)

BOB = f"lesson1/bob/{'a' * 32}"
CAROL = f"lesson1/carol/{'b' * 32}"
SPACE = {
    "dataset": "dtd",
    "lr": optunalog.Float(1e-4, 1e-2, log=True),
    "n_hidden_layers": optunalog.Int(0, 3),
    "norm": optunalog.Choice(["none", "batch"]),
}


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch) -> Path:
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    monkeypatch.setattr(optunalog, "ROOT", tmp_path)
    monkeypatch.setattr(optunalog, "STORE", tmp_path / "studies.db")
    monkeypatch.setattr(optunalog, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(optunalog, "_current", None)
    return tmp_path


def design_script(tmp_path: Path) -> Path:
    path = tmp_path / "my_design.py"
    path.write_text("import torch\n\nmodel = torch.nn.Linear(4, 2)\n")
    return path


def trial_of(study_name: str) -> optuna.trial.FrozenTrial:
    study = optuna.load_study(study_name=study_name, storage=optunalog.storage())
    (trial,) = study.trials
    return trial


def artifact(name: str, trial: optuna.trial.FrozenTrial) -> str:
    from optuna.artifacts import FileSystemArtifactStore, get_all_artifact_meta

    store = FileSystemArtifactStore(str(optunalog.ARTIFACTS))
    meta = {
        m.filename: m
        for m in get_all_artifact_meta(
            trial, storage=optuna.storages.RDBStorage(optunalog.storage())
        )
    }
    return store.open_reader(meta[name].artifact_id).read().decode()


def test_every_trial_gets_a_stable_run_id():
    optunalog.use_study("lesson1-notebook-runs")
    optunalog.log_run("a", "baseline", scores={"accuracy": [0.7, 0.8]})
    optunalog.log_run("b", "baseline", scores={"accuracy": [0.7]})
    study = optuna.load_study(
        study_name="lesson1-notebook-runs", storage=optunalog.storage()
    )
    ids = [t.user_attrs["run_id"] for t in study.trials]
    assert len(set(ids)) == 2
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)


def test_log_run_writes_the_mean_the_spread_and_the_given_params():
    optunalog.CHOICES.update(encoder=["dinov2-small"])
    optunalog.use_study("lesson1-notebook-runs")
    optunalog.log_run(
        "base",
        "baseline",
        dataset="dtd",
        params={"lr": 0.01, "encoder": "dinov2-small", "skip": True},
        scores={"accuracy": [0.7, 0.8]},
        train_seconds=1.5,
    )
    trial = trial_of("lesson1-notebook-runs")
    assert trial.value == pytest.approx(0.75)
    assert trial.params == {"lr": 0.01, "encoder": "dinov2-small", "skip": True}
    assert trial.intermediate_values == {0: 0.7, 1: 0.8}
    assert trial.user_attrs["n_repeats"] == 2
    assert trial.user_attrs["accuracy_std"] == pytest.approx(0.05)
    assert trial.user_attrs["role"] == "baseline"
    assert trial.user_attrs["dataset"] == "dtd"


def test_log_run_writes_builds_on_and_the_design_script(tmp_path):
    optunalog.use_study("lesson1-study-sam-depth-x-runs")
    optunalog.log_run(
        "mine",
        "final",
        scores={"accuracy": [0.8]},
        builds_on=[BOB, f" {CAROL} "],
        design=design_script(tmp_path),
    )
    trial = trial_of("lesson1-study-sam-depth-x-runs")
    assert trial.user_attrs["builds_on"] == f"{BOB},{CAROL}"
    assert artifact("design.py", trial) == design_script(tmp_path).read_text()


def test_builds_on_that_is_not_a_submission_ref_raises_before_anything_is_written():
    optunalog.use_study("lesson1-notebook-runs")
    with pytest.raises(ValueError, match="submission_ref"):
        optunalog.log_run(
            "mine", "final", scores={"accuracy": [0.8]}, builds_on="bob/abc"
        )
    study = optuna.load_study(
        study_name="lesson1-notebook-runs", storage=optunalog.storage()
    )
    assert study.trials == []


def test_a_design_script_over_the_limit_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(optunalog, "MAX_DESIGN_BYTES", 10)
    optunalog.use_study("lesson1-notebook-runs")
    with pytest.raises(ValueError, match="more than a submission carries"):
        optunalog.log_run(
            "mine",
            "final",
            scores={"accuracy": [0.8]},
            design=design_script(tmp_path),
        )


def test_the_architecture_is_the_note_and_an_artifact():
    optunalog.use_study("lesson1-notebook-runs")
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    optunalog.log_run(
        "m",
        "baseline",
        scores={"accuracy": [0.5]},
        model=model,
        example=torch.zeros(2, 4),
    )
    trial = trial_of("lesson1-notebook-runs")
    assert trial.user_attrs["n_params"] == 4 * 3 + 3 + 3 * 2 + 2
    assert "Linear(4->3)" in trial.user_attrs["architecture"]
    arch = json.loads(artifact("architecture.json", trial))
    assert arch["n_params"] == trial.user_attrs["n_params"]


def test_a_pruned_trial_is_recorded_with_the_epochs_it_ran():
    study = optunalog.search_study(
        "lesson1-notebook",
        SPACE,
        pruner=optuna.pruners.PercentilePruner(
            50.0, n_startup_trials=0, n_warmup_steps=0
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        config = optunalog.draw(trial, SPACE)
        try:
            for epoch in range(5):
                trial.report(1 / (trial.number + 1) + 0 * len(config), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 0.9
        finally:
            optunalog.record(
                trial, "trial", dataset="dtd", curves={"train_loss": [1.0, 0.5]}
            )

    study.optimize(objective, n_trials=4)
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.PRUNED in states
    for t in study.trials:
        assert t.user_attrs["role"] == "trial"
        assert t.user_attrs["run_id"]
        assert t.user_attrs["train_loss"] == [1.0, 0.5]
        assert t.user_attrs["config"]["dataset"] == "dtd"
        assert 1 <= len(t.intermediate_values) <= 5


def test_a_search_study_carries_its_space():
    study = optunalog.search_study(
        "lesson1-notebook", SPACE, attrs={"origin": "notebook"}
    )
    assert study.user_attrs["searched_axes"] == "lr,n_hidden_layers,norm"
    assert "dataset=dtd" in study.user_attrs["search_space"]
    assert "lr ~ logU(1e-4, 1e-2)" in study.user_attrs["search_space"]
    assert study.user_attrs["origin"] == "notebook"


def test_check_study_name_names_a_first_name_and_a_topic():
    assert (
        optunalog.check_study_name(" Sam-Depth-On-Eurosat ") == "sam-depth-on-eurosat"
    )
    for bad in ("name-your-study", "", "sam", "Sam depth"):
        with pytest.raises(ValueError):
            optunalog.check_study_name(bad)


def test_an_idea_has_an_id_and_a_name_of_its_author_words_and_parents():
    idea = optunalog.idea("Does depth  help on EuroSAT", author="Alice")
    assert idea["idea"] == "does depth help on eurosat"
    assert idea["idea_author"] == "alice"
    assert idea["idea_parents"] == ""
    assert idea["idea_name"] == optunalog.idea_name(idea["idea_id"])
    child = optunalog.idea(
        "depth and norm together",
        builds_on=["zany-heron-41", idea["idea_name"]],
        author="alice",
    )
    assert child["idea_parents"] == ",".join(
        sorted(["zany-heron-41", idea["idea_name"]])
    )
    assert child["idea_id"] != idea["idea_id"]


@pytest.mark.parametrize(
    "words, builds_on",
    [("depth", None), ("x" * 121, None), ("does depth help", "Zany Heron")],
)
def test_an_idea_that_cannot_be_named_raises(words, builds_on):
    with pytest.raises(ValueError, match="idea:"):
        optunalog.idea(words, builds_on=builds_on, author="alice")


def test_the_author_is_read_through_gh_once_and_kept(monkeypatch, store):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("Done", (), {"stdout": "Alice-1\n"})()

    monkeypatch.setattr(optunalog, "TESTING", False)
    monkeypatch.setattr(optunalog.subprocess, "run", fake_run)
    assert optunalog.idea("does depth help on eurosat")["idea_author"] == "alice-1"
    assert optunalog.login() == "alice-1"
    assert len(calls) == 1


def test_without_gh_the_error_says_what_to_pass(monkeypatch):
    def no_gh(cmd, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(optunalog, "TESTING", False)
    monkeypatch.setattr(optunalog.subprocess, "run", no_gh)
    with pytest.raises(ValueError, match="author='<login>'"):
        optunalog.idea("does depth help on eurosat")


def test_a_study_carries_its_idea_and_refuses_another():
    idea = optunalog.idea("does depth help on eurosat", author="alice")
    study = optunalog.search_study("lesson1-study-alice-depth", SPACE, idea=idea)
    assert {k: study.user_attrs[k] for k in idea} == idea
    runs = optunalog.use_study("lesson1-study-alice-depth-runs", idea=idea)
    assert runs.user_attrs["idea_name"] == idea["idea_name"]
    optunalog.search_study("lesson1-study-alice-depth", SPACE, idea=idea)

    other = optunalog.idea("does width help on eurosat", author="alice")
    with pytest.raises(ValueError, match="another idea is another study"):
        optunalog.search_study("lesson1-study-alice-depth", SPACE, idea=other)
    with pytest.raises(ValueError, match="another idea is another study"):
        optunalog.use_study("lesson1-study-alice-depth-runs", idea=other)


def test_a_submitted_study_with_an_idea_passes_the_check(tmp_path):
    """What optunalog writes is what the class repository's check recomputes."""
    from run_submission import check_idea  # ty: ignore[import-not-found]

    idea = optunalog.idea("depth and norm", builds_on="zany-heron-41", author="alice")
    study = optunalog.search_study("lesson1-study-alice-depth", SPACE, idea=idea)
    check_idea(dict(study.user_attrs))


def test_budget_turns_minutes_into_trials(monkeypatch):
    study = optunalog.search_study("lesson1-notebook", SPACE)

    def objective(trial):
        optunalog.draw(trial, {**SPACE, "epochs": 10})
        return 0.5

    study.optimize(objective, n_trials=3)
    monkeypatch.setattr(
        optuna.trial.FrozenTrial,
        "duration",
        property(lambda self: timedelta(seconds=20)),
    )
    out = optunalog.budget(10, like="lesson1-notebook", space=SPACE)
    assert out["seconds_per_trial"] == 20
    assert out["n_trials"] == 30 - 2 * 3
    assert out["searched_axes"] == 3 and out["trials_per_axis"] == 8.0
    half = optunalog.budget(10, like="lesson1-notebook", epochs=5)
    assert half["seconds_per_trial"] == 10 and half["n_trials"] == 60 - 6
    assert optunalog.budget(1, like="lesson1-notebook")["n_trials"] == 0


def test_budget_needs_a_study_that_ran():
    with pytest.raises(ValueError, match="run it first"):
        optunalog.budget(10, like="lesson1-notebook")
    optunalog.search_study("lesson1-notebook", SPACE)
    with pytest.raises(ValueError, match="no finished trial"):
        optunalog.budget(10, like="lesson1-notebook")


def test_the_idea_placeholder_passes_only_under_the_tester(monkeypatch):
    monkeypatch.setattr(optunalog, "TESTING", True)
    assert optunalog.idea(optunalog.IDEA_PLACEHOLDER)["idea_author"] == "tester"
    monkeypatch.setattr(optunalog, "TESTING", False)
    with pytest.raises(ValueError, match="in your own words"):
        optunalog.idea(optunalog.IDEA_PLACEHOLDER, author="alice")


def test_the_handed_out_idea_is_one_root_for_the_whole_class():
    root = optunalog.handed_out(1)
    assert root["idea_author"] == optunalog.COURSE and root["idea_parents"] == ""
    assert root == optunalog.handed_out(1) != optunalog.handed_out(2)
    first = optunalog.idea(
        "does depth still help", builds_on=root["idea_name"], author="sam"
    )
    assert first["idea_parents"] == root["idea_name"]


def test_a_study_refuses_trials_of_another_space(monkeypatch):
    monkeypatch.setattr(optunalog, "TESTING", False)
    name = "lesson1-study-alice-depth"
    study = optunalog.search_study(name, SPACE)
    study.optimize(lambda trial: float(len(optunalog.draw(trial, SPACE))), n_trials=2)
    optunalog.search_study(name, SPACE)  # the same space adds to it
    other = {**SPACE, "extra_axis": optunalog.Choice([False, True])}
    with pytest.raises(ValueError, match="trials of another search space"):
        optunalog.search_study(name, other, attrs={"study": "overwritten"})
    kept = optunalog.search_study(name, SPACE)
    assert len(kept.trials) == 2 and "study" not in kept.user_attrs


def test_under_the_tester_a_study_of_another_space_or_idea_starts_again(monkeypatch):
    monkeypatch.setattr(optunalog, "TESTING", True)
    name = "lesson1-study-name-your-study"
    first = optunalog.idea("an idea of the first execution", author="alice")
    study = optunalog.search_study(name, SPACE, idea=first)
    study.optimize(lambda trial: float(len(optunalog.draw(trial, SPACE))), n_trials=2)
    second = optunalog.idea("an idea of the second execution", author="alice")
    again = optunalog.search_study(name, SPACE, idea=second)
    assert again.trials == [] and again.user_attrs["idea_name"] == second["idea_name"]
