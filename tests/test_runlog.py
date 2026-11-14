"""runlog against a temporary sqlite store."""

import sys
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "notebooks/1_pytorch_intro")
)

import runlog  # ty: ignore[import-not-found]  (the lesson folder is put on sys.path above)

BOB = f"lesson1/bob/{'a' * 32}"
CAROL = f"lesson1/carol/{'b' * 32}"


@pytest.fixture
def store(tmp_path, monkeypatch) -> MlflowClient:
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    runlog.use_experiment("lesson1-notebook")
    return MlflowClient(tracking_uri=uri)


def design_script(tmp_path: Path) -> Path:
    path = tmp_path / "my_design.py"
    path.write_text("import torch\n\nmodel = torch.nn.Linear(4, 2)\n")
    return path


def artifact_text(client: MlflowClient, run_id: str, name: str, tmp: Path) -> str:
    return Path(client.download_artifacts(run_id, name, str(tmp))).read_text()


def test_log_run_writes_builds_on_and_the_design_script(store, tmp_path):
    design = design_script(tmp_path)
    run_id = runlog.log_run(
        "mine", "final", scores={"accuracy": [0.8]}, builds_on=BOB, design=design
    )
    run = store.get_run(run_id)
    assert run.data.tags["builds_on"] == BOB
    names = [a.path for a in store.list_artifacts(run_id)]
    assert "design.py" in names
    out = tmp_path / "out"
    out.mkdir()
    assert artifact_text(store, run_id, "design.py", out) == design.read_text()


def test_study_log_run_writes_several_builds_on(store, tmp_path):
    study = runlog.Study("sam-depth-on-eurosat", space={"lr": 0.01}, budget={})
    design = design_script(tmp_path)
    run_id = study.log_run(
        "mine", "trial", builds_on=[BOB, f" {CAROL} "], design=str(design)
    )
    run = store.get_run(run_id)
    assert run.data.tags["builds_on"] == f"{BOB},{CAROL}"
    names = {a.path for a in store.list_artifacts(run_id)}
    assert names == {"design.py", "search_space.json"}


def test_without_them_neither_is_written(store):
    run_id = runlog.log_run("plain", "baseline", scores={"accuracy": [0.5]})
    assert "builds_on" not in store.get_run(run_id).data.tags
    assert store.list_artifacts(run_id) == []


def logged_runs(client: MlflowClient) -> list:
    exp = client.get_experiment_by_name("lesson1-notebook")
    assert exp is not None
    return client.search_runs([exp.experiment_id])


def test_a_string_of_refs_is_split_stripped_and_emptied(store):
    run_id = runlog.log_run(
        "mine", "final", scores={"accuracy": [0.8]}, builds_on=f" {BOB}, {CAROL},"
    )
    assert store.get_run(run_id).data.tags["builds_on"] == f"{BOB},{CAROL}"


@pytest.mark.parametrize(
    "bad", [f"{BOB}.json", f"lesson1/bob_/{'a' * 32}", "bob", [BOB, "lesson1/bob"]]
)
def test_a_bad_ref_raises_before_anything_is_logged(store, bad):
    with pytest.raises(ValueError, match="is not a submission_ref"):
        runlog.log_run("mine", "final", scores={"accuracy": [0.8]}, builds_on=bad)
    assert logged_runs(store) == []
