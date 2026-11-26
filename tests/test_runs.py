"""hpo.record.runs, hpo.record.ideas, hpo.search.budget and hpo.search.space against a temporary sqlite store."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import optuna
import pytest
import torch
from torch import nn

from hpo.measure import Architecture, Confusion, Curves, Repeats, Scalars
from hpo.record import DesignScript, RunRecord, Store, StudyLog
from hpo.record import store as store_module
from hpo.record.ideas import IDEA_PLACEHOLDER, Idea, study_name
from hpo.record.protocol import COURSE, idea_name
from hpo.search import Budget, Choice, Float, Int, Space

BOB = f"lesson1/bob/{'a' * 32}"
CAROL = f"lesson1/carol/{'b' * 32}"
SPACE = Space(
    {
        "dataset": "dtd",
        "lr": Float(1e-4, 1e-2, log=True),
        "n_hidden_layers": Int(0, 3),
        "norm": Choice(["none", "batch"]),
    }
)


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    monkeypatch.delenv(store_module.TESTER, raising=False)


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path)


def design_script(tmp_path: Path) -> Path:
    path = tmp_path / "my_design.py"
    path.write_text("import torch\n\nmodel = torch.nn.Linear(4, 2)\n", encoding="utf-8")
    return path


def only_trial(log: StudyLog) -> optuna.trial.FrozenTrial:
    (trial,) = log.trials()
    return trial


def artifact(store: Store, name: str, trial: optuna.trial.FrozenTrial) -> str:
    from optuna.artifacts import FileSystemArtifactStore, get_all_artifact_meta

    files = FileSystemArtifactStore(str(store.artifacts))
    meta = {
        m.filename: m
        for m in get_all_artifact_meta(
            trial, storage=optuna.storages.RDBStorage(store.url)
        )
    }
    return files.open_reader(meta[name].artifact_id).read().decode()


def confusion(right_n: int, right_f: int) -> Confusion:
    """A confusion over the classes N (10 rows) and F (10 rows)."""
    return Confusion(
        classes=["N", "F"], counts=[[right_n, 10 - right_n], [10 - right_f, right_f]]
    )


def test_every_trial_gets_a_stable_run_id(store):
    log = StudyLog.runs("lesson1-notebook-runs", store=store)
    log.log("a", "baseline", Scalars(accuracy=[0.7, 0.8]))
    log.log("b", "baseline", Scalars(accuracy=[0.7]))
    ids = [t.user_attrs["run_id"] for t in log.trials()]
    assert len(set(ids)) == 2
    assert all(len(i) == 32 and int(i, 16) >= 0 for i in ids)


def test_a_run_writes_the_mean_the_spread_and_the_given_params(store):
    log = StudyLog.runs(
        "lesson1-notebook-runs", choices={"encoder": ["dinov2-small"]}, store=store
    )
    log.log(
        "base",
        "baseline",
        Scalars(accuracy=[0.7, 0.8], seconds=1.5),
        dataset="dtd",
        params={"lr": 0.01, "encoder": "dinov2-small", "skip": True, "head": "mlp"},
    )
    trial = only_trial(log)
    assert trial.value == pytest.approx(0.75)
    assert trial.params == {"lr": 0.01, "encoder": "dinov2-small", "skip": True}
    assert trial.user_attrs["head"] == "mlp"  # a string without choices is an attr
    assert trial.intermediate_values == {0: 0.7, 1: 0.8}
    assert trial.user_attrs["n_repeats"] == 2
    assert trial.user_attrs["accuracy_std"] == pytest.approx(0.05)
    assert trial.user_attrs["train_seconds"] == 1.5
    assert (trial.user_attrs["role"], trial.user_attrs["dataset"]) == (
        "baseline",
        "dtd",
    )


def test_a_run_writes_builds_on_and_the_design_script(store, tmp_path):
    log = StudyLog.runs("lesson1-study-sam-depth-x-runs", store=store)
    log.log(
        "mine",
        "final",
        Scalars(accuracy=[0.8]),
        builds_on=[BOB, f" {CAROL} "],
        design=DesignScript.read(design_script(tmp_path)),
    )
    trial = only_trial(log)
    assert trial.user_attrs["builds_on"] == f"{BOB},{CAROL}"
    assert artifact(store, "design.py", trial) == design_script(tmp_path).read_text(
        encoding="utf-8"
    )


def test_builds_on_that_is_not_a_submission_ref_raises_before_anything_is_written(
    store,
):
    log = StudyLog.runs("lesson1-notebook-runs", store=store)
    with pytest.raises(ValueError, match="submission_ref"):
        log.log("mine", "final", Scalars(accuracy=[0.8]), builds_on="bob/abc")
    assert log.trials() == []


def test_a_design_script_over_the_limit_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("hpo.record.runs.MAX_ARTIFACT_BYTES", 10)
    with pytest.raises(ValueError, match="more than a submission carries"):
        DesignScript.read(design_script(tmp_path))


def test_the_architecture_is_the_note_and_an_artifact(store):
    log = StudyLog.runs("lesson1-notebook-runs", store=store)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    log.log(
        "m",
        "baseline",
        Scalars(accuracy=[0.5]),
        architecture=Architecture.of(model, torch.zeros(2, 4)),
    )
    trial = only_trial(log)
    assert trial.user_attrs["n_params"] == 4 * 3 + 3 + 3 * 2 + 2
    assert "Linear(4->3)" in trial.user_attrs["architecture"]
    assert trial.user_attrs["flops_per_image"] == 2 * (4 * 3 + 3 * 2)
    assert trial.user_attrs["latency_us"] > 0
    arch = json.loads(artifact(store, "architecture.json", trial))
    assert arch["n_params"] == trial.user_attrs["n_params"]
    assert arch["flops_per_image"] == trial.user_attrs["flops_per_image"]


def test_the_architecture_of_a_model_with_two_streams():
    class Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(6, 2)

        def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return self.head(torch.cat(x, dim=-1))

    arch = Architecture.of(Two(), (torch.zeros(2, 4), torch.zeros(2, 2)))
    assert arch.n_params == 6 * 2 + 2 and "Linear(6->2)" in arch.signature


def test_a_pruned_trial_is_recorded_with_the_epochs_it_ran(store):
    log = StudyLog.search(
        "lesson1-notebook",
        SPACE,
        pruner=optuna.pruners.PercentilePruner(
            50.0, n_startup_trials=0, n_warmup_steps=0
        ),
        store=store,
    )

    def objective(trial: optuna.Trial) -> float:
        config = SPACE.draw(trial)
        curves = Curves()
        try:
            for epoch in range(5):
                curves.add(1.0 / (epoch + 1), 1.0, None)
                trial.report(1 / (trial.number + 1), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return 0.9
        finally:
            log.record(
                trial,
                RunRecord(
                    role="trial",
                    dataset="dtd",
                    config=config,
                    fixed=SPACE.fixed(),
                    curves=curves,
                ),
            )

    log.optimize(objective, 4)
    assert log.trials(optuna.trial.TrialState.PRUNED)
    for t in log.trials():
        assert t.user_attrs["role"] == "trial" and t.user_attrs["run_id"]
        assert t.user_attrs["train_loss"][0] == 1.0
        assert t.user_attrs["config"]["dataset"] == "dtd"
        assert t.user_attrs["dataset"] == "dtd"  # a fixed value as its own attr
        assert 1 <= len(t.intermediate_values) <= 5


def test_a_search_study_carries_its_space(store):
    log = StudyLog.search(
        "lesson1-notebook", SPACE, attrs={"origin": "notebook"}, store=store
    )
    attrs = log.study.user_attrs
    assert attrs["searched_axes"] == "lr,n_hidden_layers,norm"
    assert "dataset=dtd" in attrs["search_space"]
    assert "lr ~ logU(1e-4, 1e-2)" in attrs["search_space"]
    assert attrs["origin"] == "notebook"


def test_a_space_draws_its_axes_and_keeps_its_fixed_values():
    study = optuna.create_study()
    trial = study.ask()
    config = SPACE.draw(trial)
    assert config["dataset"] == "dtd" and config["n_hidden_layers"] in range(4)
    assert SPACE.fixed() == {"dataset": "dtd"}
    assert Space(dict(SPACE)) == SPACE != Space({"dataset": "eurosat"})


def test_a_study_name_names_a_first_name_and_a_topic():
    assert study_name(" Sam-Depth-On-Eurosat ") == "sam-depth-on-eurosat"
    for bad in ("name-your-study", "", "sam", "Sam depth"):
        with pytest.raises(ValueError):
            study_name(bad)


def test_an_idea_has_an_id_and_a_name_of_its_author_words_and_parents():
    idea = Idea.of("Does depth  help on EuroSAT", author="Alice")
    assert (idea.words, idea.author, idea.parents) == (
        "does depth help on eurosat",
        "alice",
        (),
    )
    assert idea.name == idea_name(idea.id)
    child = Idea.of(
        "depth and norm together",
        builds_on=["zany-heron-41", idea.name],
        author="alice",
    )
    assert child.attrs()["idea_parents"] == ",".join(
        sorted(["zany-heron-41", idea.name])
    )
    assert child.id != idea.id


@pytest.mark.parametrize(
    "words, builds_on",
    [("depth", None), ("x" * 121, None), ("does depth help", "Zany Heron")],
)
def test_an_idea_that_cannot_be_named_raises(words, builds_on):
    with pytest.raises(ValueError, match="idea:"):
        Idea.of(words, builds_on=builds_on, author="alice")


def test_the_author_is_read_through_gh_once_and_kept(monkeypatch, store):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return type("Done", (), {"stdout": "Alice-1\n"})()

    monkeypatch.setattr(store_module.subprocess, "run", fake_run)
    assert Idea.of("does depth help on eurosat", store=store).author == "alice-1"
    assert store.login() == "alice-1"
    assert len(calls) == 1


def test_without_gh_the_error_says_what_to_pass(monkeypatch, store):
    def no_gh(cmd, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(store_module.subprocess, "run", no_gh)
    with pytest.raises(ValueError, match="author='<login>'"):
        Idea.of("does depth help on eurosat", store=store)


def test_a_study_carries_its_idea_and_refuses_another(store):
    idea = Idea.of("does depth help on eurosat", author="alice")
    log = StudyLog.search("lesson1-study-alice-depth", SPACE, idea, store=store)
    assert {k: log.study.user_attrs[k] for k in idea.attrs()} == idea.attrs()
    runs = StudyLog.runs("lesson1-study-alice-depth-runs", idea, store=store)
    assert runs.study.user_attrs["idea_name"] == idea.name
    StudyLog.search("lesson1-study-alice-depth", SPACE, idea, store=store)

    other = Idea.of("does width help on eurosat", author="alice")
    with pytest.raises(ValueError, match="another idea is another study"):
        StudyLog.search("lesson1-study-alice-depth", SPACE, other, store=store)
    with pytest.raises(ValueError, match="another idea is another study"):
        StudyLog.runs("lesson1-study-alice-depth-runs", other, store=store)


def test_a_submitted_study_with_an_idea_passes_the_check(store):
    """What hpo writes is what the class repository's check recomputes."""
    from run_submission import check_idea  # ty: ignore[import-not-found]

    idea = Idea.of("depth and norm", builds_on="zany-heron-41", author="alice")
    log = StudyLog.search("lesson1-study-alice-depth", SPACE, idea, store=store)
    check_idea(dict(log.study.user_attrs))


def test_a_budget_turns_minutes_into_trials(monkeypatch, store):
    log = StudyLog.search("lesson1-notebook", SPACE, store=store)
    with_epochs = Space({**SPACE, "epochs": 10})

    def objective(trial: optuna.Trial) -> float:
        log.record(trial, RunRecord(role="trial", config=with_epochs.draw(trial)))
        return 0.5

    log.optimize(objective, 3)
    monkeypatch.setattr(
        optuna.trial.FrozenTrial,
        "duration",
        property(lambda self: timedelta(seconds=20)),
    )
    out = Budget.at(10, log.seconds_per_trial(), log.name, space=SPACE)
    assert out.seconds_per_trial == 20 and out.measured_on == "lesson1-notebook"
    assert out.n_trials == 30 - 2 * 3
    assert out.searched_axes == 3 and out.trials_per_axis == 8.0
    half = Budget.at(10, log.seconds_per_trial(5), log.name, epochs=5)
    assert half.seconds_per_trial == 10 and half.n_trials == 60 - 6
    assert Budget.at(1, log.seconds_per_trial(), log.name).n_trials == 0


def test_a_budget_needs_a_study_that_ran(store):
    with pytest.raises(ValueError, match="run it first"):
        StudyLog.load("lesson1-notebook", store)
    StudyLog.search("lesson1-notebook", SPACE, store=store)
    with pytest.raises(ValueError, match="no finished trial"):
        StudyLog.load("lesson1-notebook", store).seconds_per_trial()


def test_the_idea_placeholder_passes_only_under_the_tester(monkeypatch):
    monkeypatch.setenv(store_module.TESTER, "1")
    assert Idea.of(IDEA_PLACEHOLDER).author == "tester"
    monkeypatch.delenv(store_module.TESTER)
    with pytest.raises(ValueError, match="in your own words"):
        Idea.of(IDEA_PLACEHOLDER, author="alice")


def test_the_handed_out_idea_is_one_root_for_the_whole_class():
    root = Idea.handed_out(1)
    assert root.author == COURSE and root.parents == ()
    assert root == Idea.handed_out(1) != Idea.handed_out(2)
    first = Idea.of("does depth still help", builds_on=root.name, author="sam")
    assert first.parents == (root.name,)


def test_a_study_refuses_trials_of_another_space(store):
    name = "lesson1-study-alice-depth"
    log = StudyLog.search(name, SPACE, store=store)
    log.optimize(lambda trial: float(len(SPACE.draw(trial))), 2)
    StudyLog.search(name, SPACE, store=store)  # the same space adds to it
    other = Space({**SPACE, "extra_axis": Choice([False, True])})
    with pytest.raises(ValueError, match="trials of another search space"):
        StudyLog.search(name, other, attrs={"study": "overwritten"}, store=store)
    kept = StudyLog.search(name, SPACE, store=store)
    assert len(kept.trials()) == 2 and "study" not in kept.study.user_attrs


def test_under_the_tester_a_study_of_another_idea_starts_again(tmp_path):
    store = Store(tmp_path, testing=True)
    name = "lesson1-study-name-your-study"
    first = Idea.of("an idea of the first execution", author="alice")
    log = StudyLog.search(name, SPACE, first, store=store)
    log.optimize(lambda trial: float(len(SPACE.draw(trial))), 2)
    second = Idea.of("an idea of the second execution", author="alice")
    again = StudyLog.search(name, SPACE, second, store=store)
    assert again.trials() == []
    assert again.study.user_attrs["idea_name"] == second.name
    assert store.db.name == "test.db"


def test_a_run_logs_every_metric_and_its_value_is_the_studys_objective(store):
    log = StudyLog.runs(
        "lesson2-study-sam-gate-runs", objective="macro_recall", store=store
    )
    log.log(
        "winner", "final", Repeats(runs=[confusion(9, 2), confusion(9, 4)], seconds=1)
    )
    trial = only_trial(log)
    assert log.study.user_attrs["objective"] == "macro_recall"
    assert log.study.metric_names == ["macro_recall"]
    assert trial.value == pytest.approx(0.6)
    assert trial.intermediate_values == {0: pytest.approx(0.55), 1: pytest.approx(0.65)}
    assert trial.user_attrs["metrics"]["recall"]["F"] == pytest.approx(0.3)
    assert trial.user_attrs["metrics_std"]["recall"]["F"] == pytest.approx(0.1)
    assert trial.user_attrs["macro_recall_std"] == pytest.approx(0.05)
    assert trial.user_attrs["confusion"] == {
        "classes": ["N", "F"],
        "counts": [[18, 2], [14, 6]],
    }


def test_a_study_refuses_another_objective(store):
    log = StudyLog.runs("lesson2-study-sam-gate-runs", store=store)
    log.log("base", "baseline", Scalars(accuracy=[0.8]))
    with pytest.raises(ValueError, match="objective 'accuracy', not 'macro_recall'"):
        StudyLog.runs(
            "lesson2-study-sam-gate-runs", objective="macro_recall", store=store
        )


def test_an_objective_that_is_no_metric_is_refused(store):
    with pytest.raises(ValueError, match="objective must be one of"):
        StudyLog.runs("lesson2-study-x-runs", objective="loss", store=store)  # ty: ignore[invalid-argument-type]


def test_a_run_that_does_not_log_the_objective_is_refused(store):
    log = StudyLog.runs(
        "lesson2-study-sam-gate-runs", objective="macro_recall", store=store
    )
    with pytest.raises(ValueError, match="does not log"):
        log.log("base", "baseline", Scalars(accuracy=[0.8]))


def test_a_study_on_macro_recall_passes_the_check(store, tmp_path):
    import submit_runs  # ty: ignore[import-not-found]
    from run_submission import check  # ty: ignore[import-not-found]

    log = StudyLog.runs(
        "lesson2-study-sam-gate-runs",
        Idea.of("does a gate find f", author="sam"),
        objective="macro_recall",
        store=store,
    )
    log.log(
        "winner", "final", Repeats(runs=[confusion(9, 2), confusion(9, 4)], seconds=1)
    )
    found_in = submit_runs.Store(store.db)
    (found,) = submit_runs.select_runs(found_in, 2, None, None)
    code = {"commit": None, "dirty": None, "remote": None}
    sub = submit_runs.to_submission(found_in, found, 2, "sam", code, datetime.now(UTC))
    assert sub.objective() == "macro_recall"
    assert sub.metric("macro_recall") == pytest.approx(0.6)
    assert sub.metric("recall/F") == pytest.approx(0.3)
    assert sub.spread("recall/F") == pytest.approx(0.1)
    out = tmp_path / "class"
    (out / sub.relpath()).parent.mkdir(parents=True)
    (out / sub.relpath()).write_text(sub.dump(), encoding="utf-8")
    assert check(out, [sub.relpath()], "sam", set()) == []
