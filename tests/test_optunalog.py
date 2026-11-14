"""optunalog against a temporary sqlite store."""

import json
import sys
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
