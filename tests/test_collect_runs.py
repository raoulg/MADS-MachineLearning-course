import sys

import collect_runs  # ty: ignore[import-not-found]
import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def test_a_failed_import_is_reported_removed_and_fails_the_run(
    tmp_path, submission, monkeypatch, capsys
):
    repo = tmp_path / "runs"
    (repo / ".git").mkdir(parents=True)
    good, bad = submission("alice"), submission("bob")
    for sub in (good, bad):
        path = repo / sub.relpath()
        path.parent.mkdir(parents=True)
        path.write_text(sub.dump())

    fill = collect_runs.fill_run

    def fail_for_bob(client, rid, sub, tags):
        if sub.student == "bob":
            raise MlflowException("disk full")
        fill(client, rid, sub, tags)

    monkeypatch.setattr(collect_runs, "fill_run", fail_for_bob)
    uri = f"sqlite:///{tmp_path / 'class.db'}"
    argv = ["collect_runs.py", "--runs-repo", str(repo), "--tracking-uri", uri]
    monkeypatch.setattr(sys, "argv", [*argv, "--no-pull"])
    with pytest.raises(SystemExit) as e:
        collect_runs.main()
    assert e.value.code != 0
    out = capsys.readouterr().out
    assert f"failed {bad.relpath()}: disk full" in out
    assert "1 failed" in out and "1 new" in out

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("lesson1-notebook")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert [r.data.tags["student"] for r in runs] == ["alice"]
