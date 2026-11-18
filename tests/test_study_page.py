"""`scripts/study_page.py` on a made-up store of two studies, the second built on the
first; nothing here trains, downloads or runs atlas."""

from pathlib import Path

import optuna
import pytest
import study_page  # ty: ignore[import-not-found]  (scripts/ is on sys.path, see conftest.py)

from hpo.measure import Confusion, Repeats
from hpo.record import DesignScript, Idea, RunRecord, Store, StudyLog
from hpo.record import store as store_module
from hpo.search import Choice, Float, Space

SPACE = Space(
    {
        "dataset": "dtd",
        "lr": Float(1e-4, 1e-2, log=True),
        "skip": Choice([False, True]),
        "epochs": 3,
    }
)
DESIGN = """
from torch import Tensor, nn
from jaxtyping import Float
from vectormesh.design import option


class Head(nn.Module):
    def hidden(self, x: Float[Tensor, "batch {self.dim}"]) -> Float[Tensor, "batch 8"]:
        return x

    @option("skip")
    def skip(self, x: Float[Tensor, "batch {self.dim}"]) -> Float[Tensor, "batch 8"]:
        return x

    def forward(self, x: Float[Tensor, "batch {self.dim}"]) -> Float[Tensor, "batch 2"]:
        h = self.hidden(x)
        if self.config.skip:
            h = h + self.skip(x)
        return h
"""
FIRST = ("does a skip help on dtd", "a skip adds a point")
SECOND = ("a wider learning rate range with a skip", "the best lr moves up")


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    monkeypatch.delenv(store_module.TESTER, raising=False)


def repeats(right: list[int]) -> Repeats:
    """Repeats over the classes N and F, 10 rows each: `right` rows of each right."""
    return Repeats(
        runs=[
            Confusion(classes=["N", "F"], counts=[[r, 10 - r], [10 - r, r]])
            for r in right
        ],
        seconds=1.0,
    )


def study(store: Store, name: str, idea: Idea, expects: str, values, design: Path):
    """A search of `values`, one trial each, and its baseline and winner."""
    search = StudyLog.search(
        f"lesson1-study-{name}",
        SPACE,
        idea,
        attrs={"origin": "study", "study": name, "expects": expects},
        store=store,
    )
    scores = iter(values)

    def objective(trial: optuna.Trial) -> float:
        config = SPACE.draw(trial)
        value = next(scores)
        for epoch in range(3):
            trial.report(value * (epoch + 1) / 3, epoch)
        search.record(
            trial,
            RunRecord(
                role="trial",
                dataset="dtd",
                config=config,
                fixed=SPACE.fixed(),
                train_seconds=2.0 + value,
            ),
        )
        return value

    search.optimize(objective, len(values))
    runs = StudyLog.runs(f"lesson1-study-{name}-runs", idea, store=store)
    script = DesignScript.read(design)
    runs.log("baseline", "baseline", repeats([6, 7, 6]), dataset="dtd", design=script)
    runs.log("winner", "final", repeats([8, 9, 8]), dataset="dtd", design=script)


@pytest.fixture
def store(tmp_path) -> Store:
    """Alice's store: the second study is run first and builds on the first, so only
    `builds_on` can put the first in front."""
    store = Store(tmp_path / "optuna")
    store.root.mkdir()
    (store.root / "login").write_text("alice\n")
    design = tmp_path / "designs.py"
    design.write_text(DESIGN)
    handed = Idea.handed_out(1).name
    first = Idea.of(FIRST[0], builds_on=handed, author="alice")
    second = Idea.of(SECOND[0], builds_on=first.name, author="alice")
    study(store, "alice-lr-range", second, SECOND[1], [0.7, 0.8, 0.75], design)
    study(store, "alice-skip", first, FIRST[1], [0.6, 0.65], design)
    return store


@pytest.fixture
def folder(tmp_path) -> Path:
    out = tmp_path / "exercises"
    out.mkdir()
    return out


def page(store: Store, folder: Path) -> str:
    subs, login, source = study_page.read_store(store, 1)
    own = study_page.read_own(folder, 1)
    design = study_page.design_tab(folder, which=lambda _: None)
    studies = study_page.studies_of(subs)
    return study_page.build_page(studies, 1, login, source, own, folder, design)


def test_both_ideas_and_their_expectations_are_on_the_page(store, folder):
    text = page(store, folder)
    for words, expects in (FIRST, SECOND):
        assert words in text and expects in text
    assert "Lesson 1: what alice studied" in text
    for tab in ("Overview", "Design", "alice-skip", "alice-lr-range"):
        assert f">{tab}</button>" in text


def test_the_line_of_ideas_follows_builds_on(store, folder):
    text = page(store, folder)
    line = text[text.index("class='ideas'") :]
    assert line.index(FIRST[0]) < line.index(SECOND[0])
    first = Idea.of(FIRST[0], builds_on=Idea.handed_out(1).name, author="alice")
    assert f"builds on {first.name} (alice-skip)" in text
    subs, _, _ = study_page.read_store(store, 1)
    assert [s.label for s in study_page.studies_of(subs)] == [
        "alice-skip",
        "alice-lr-range",
    ]


def test_a_study_box_gives_its_gain_in_points_and_sigmas(store, folder):
    subs, _, _ = study_page.read_store(store, 1)
    first = study_page.studies_of(subs)[0]
    points, sigma, word = first.gain()
    assert points == pytest.approx(100 * (25 / 30 - 19 / 30))
    assert sigma > 2 and word == "better"
    assert "more than two sigma above its baseline" in page(store, folder)


def test_an_empty_page_toml_gives_open_slots_and_a_filled_one_its_text(store, folder):
    text = page(store, folder)  # no page.toml yet: it is made, with empty keys
    made = (folder / "page.toml").read_text()
    for key in study_page.OWN:
        assert f'{key} = ""' in made
        assert f"page.toml · {key}" in text
    assert "Say in three sentences what you did." in text
    assert "class='slot filled'" not in text

    (folder / "page.toml").write_text(
        made.replace(
            'message = ""', 'message = "A skip helps on dtd. Lr does not."'
        ).replace('next = ""', 'next = "Try a gate."')
    )
    text = page(store, folder)
    assert "A skip helps on dtd. Lr does not." in text
    assert "Try a gate." in text
    assert "Say in three sentences what you did." not in text
    assert "What surprised you, against what you expected?" in text
    assert (folder / "page.toml").read_text().count("A skip helps") == 1


def test_a_missing_atlas_does_not_fail_and_says_where_to_get_it(folder):
    text = study_page.design_tab(folder, which=lambda _: None)
    assert "Atlas is not on this machine" in text
    assert "docs/setup/tooling-setup.md" in text


def test_the_architecture_comes_from_the_design_script(store, folder):
    text = page(store, folder)
    assert "<b>hidden</b>" in text and "<b>skip</b>" in text
    assert "@option(&quot;skip&quot;)" in text


def test_a_study_without_a_finished_trial_says_so(tmp_path, folder):
    store = Store(tmp_path / "optuna")
    store.root.mkdir()
    (store.root / "login").write_text("alice\n")
    idea = Idea.of(FIRST[0], builds_on=Idea.handed_out(1).name, author="alice")
    search = StudyLog.search("lesson1-study-alice-empty", SPACE, idea, store=store)

    def pruned(trial: optuna.Trial) -> float:
        search.record(trial, RunRecord(role="trial", config=SPACE.draw(trial)))
        raise optuna.TrialPruned()

    search.optimize(pruned, 2)
    text = page(store, folder)
    assert "No finished trial yet." in text
