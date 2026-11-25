"""`scripts/study_page.py` on a made-up store of two studies, the second built on the
first; nothing here trains, downloads or runs atlas."""

import json
import re
import threading
import urllib.error
import urllib.request
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


def study(
    store: Store,
    name: str,
    idea: Idea,
    expects: str,
    values,
    design: Path,
    baseline: list[int],
    final: list[int],
):
    """A search of `values`, one trial each, and its baseline and winner, whose repeats
    get `baseline` and `final` rows right."""
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
    runs.log("baseline", "baseline", repeats(baseline), dataset="dtd", design=script)
    runs.log("winner", "final", repeats(final), dataset="dtd", design=script)


@pytest.fixture
def store(tmp_path) -> Store:
    """Alice's store: the second study is run first and builds on the first, so only
    `builds_on` can put the first in front."""
    store = Store(tmp_path / "optuna")
    store.root.mkdir()
    (store.root / "login").write_text("alice\n", encoding="utf-8")
    design = tmp_path / "designs.py"
    design.write_text(DESIGN, encoding="utf-8")
    handed = Idea.handed_out(1).name
    first = Idea.of(FIRST[0], builds_on=handed, author="alice")
    second = Idea.of(SECOND[0], builds_on=first.name, author="alice")
    lr = [0.7, 0.8, 0.75]
    study(store, "alice-lr-range", second, SECOND[1], lr, design, [5, 6, 7], [6, 7, 7])
    study(
        store, "alice-skip", first, FIRST[1], [0.6, 0.65], design, [6, 7, 6], [8, 9, 8]
    )
    return store


@pytest.fixture
def folder(tmp_path) -> Path:
    out = tmp_path / "exercises"
    out.mkdir()
    return out


def page(store: Store, folder: Path, **kw) -> str:
    return study_page.page_of(1, folder, store=store, which=lambda _: None, **kw)[0]


def test_both_ideas_and_their_expectations_are_on_the_page(store, folder):
    text = page(store, folder)
    for words, expects in (FIRST, SECOND):
        assert words in text and expects in text
    assert "Lesson 1: what alice studied" in text
    for tab in ("Overview", "Studies", "Design"):
        assert f">{tab}</button>" in text
    assert ">Starred <span class='count'></span></button>" in text
    for slug, short in (
        ("study-alice-skip", "skip"),
        ("study-alice-lr-range", "lr range"),
    ):
        assert f"data-study='{slug}' title='{slug[6:]}'>{short}</a>" in text
        assert f"id='view-{slug}'" in text


def test_the_line_of_ideas_follows_builds_on(store, folder):
    text = page(store, folder)
    # the second idea builds on the first: a row below it, joined to it
    first = text.index("class='node' data-study='study-alice-skip' data-parent=''")
    second = text.index(
        "class='node' data-study='study-alice-lr-range' data-parent='study-alice-skip'"
    )
    assert first < second
    assert text.count("class='edge'") == 1
    handed = Idea.handed_out(1).name
    assert f"builds on {handed}, not on this page" in text
    subs, _, _ = study_page.read_store(store, 1)
    assert [s.label for s in study_page.studies_of(subs)] == [
        "alice-skip",
        "alice-lr-range",
    ]


def test_each_study_box_gives_its_own_gain_in_points_and_sigmas(store, folder):
    """By hand, with s the standard deviation over the repeats (ddof=0) and the standard
    error sqrt(s_b²/(n_b−1) + s_f²/(n_f−1)):
    alice-skip: 0.633 against 0.833, s² = 1/450 on both sides, se = 0.0471, 4.2σ;
    alice-lr-range: 0.600 against 0.667, s² = 1/150 and 1/450, se = 0.0667, 1.0σ."""
    subs, _, _ = study_page.read_store(store, 1)
    skip, lr = study_page.studies_of(subs)
    points, sigma, word = skip.gain()
    assert points == pytest.approx(100 * (25 / 30 - 19 / 30))
    assert f"{sigma:.1f}" == "4.2" and word == "better"
    points, sigma, word = lr.gain()
    # the store keeps scores to four decimals
    assert points == pytest.approx(100 * (20 / 30 - 18 / 30), abs=0.01)
    assert f"{sigma:.1f}" == "1.0" and word == "inside"
    text = page(store, folder)
    # the standard errors of the differences: 4.71 and 6.67 points
    assert "+20.0 ± 4.7 pt · clearly above its baseline: more than two" in text
    assert "+6.7 ± 6.7 pt · not measurably different from its baseline: within" in text
    assert "σ" not in text


def test_an_empty_page_toml_gives_open_slots_and_a_filled_one_its_text(store, folder):
    text = page(store, folder)  # no page.toml yet: it is made, with empty keys
    made = (folder / "page.toml").read_text(encoding="utf-8")
    for key in study_page.OWN:
        assert f'{key} = ""' in made
        assert f"page.toml · {key}" in text
    assert "Say in three sentences what you did." in text
    assert "class='slot filled'" not in text

    (folder / "page.toml").write_text(
        made.replace(
            'message = ""', 'message = "A skip helps on dtd. Lr does not."'
        ).replace('next = ""', 'next = "Try a gate."'),
        encoding="utf-8",
    )
    before = (
        (folder / "page.toml").read_bytes(),
        (folder / "page.toml").stat().st_mtime_ns,
    )
    text = page(store, folder)
    after = (
        (folder / "page.toml").read_bytes(),
        (folder / "page.toml").stat().st_mtime_ns,
    )
    assert after == before  # a page.toml that exists is never written
    assert "A skip helps on dtd. Lr does not." in text
    assert "Try a gate." in text
    assert "Say in three sentences what you did." not in text
    assert "What surprised you, against what you expected?" in text


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
    (store.root / "login").write_text("alice\n", encoding="utf-8")
    idea = Idea.of(FIRST[0], builds_on=Idea.handed_out(1).name, author="alice")
    search = StudyLog.search("lesson1-study-alice-empty", SPACE, idea, store=store)

    def pruned(trial: optuna.Trial) -> float:
        search.record(trial, RunRecord(role="trial", config=SPACE.draw(trial)))
        raise optuna.TrialPruned()

    search.optimize(pruned, 2)
    text = page(store, folder)
    assert "No finished trial yet." in text


def test_a_search_over_two_datasets_is_drawn_per_dataset(tmp_path, folder):
    store = Store(tmp_path / "optuna")
    store.root.mkdir()
    (store.root / "login").write_text("alice\n", encoding="utf-8")
    space = Space(
        {"dataset": Choice(["dtd", "eurosat"]), "lr": Float(1e-4, 1e-2, log=True)}
    )
    idea = Idea.of(FIRST[0], builds_on=Idea.handed_out(1).name, author="alice")
    search = StudyLog.search(
        "lesson1-study-alice-two",
        space,
        idea,
        sampler=optuna.samplers.RandomSampler(seed=3),
        store=store,
    )

    def objective(trial: optuna.Trial) -> float:
        config = space.draw(trial)
        # eurosat scores far higher: a picture over both would rank by dataset
        value = (0.95 if config["dataset"] == "eurosat" else 0.6) + trial.number / 100
        trial.report(value, 0)
        search.record(
            trial,
            RunRecord(role="trial", dataset=str(config["dataset"]), config=config),
        )
        return value

    search.optimize(objective, 8)
    drawn = {t.user_attrs["dataset"] for t in search.trials()}
    assert drawn == {"dtd", "eurosat"}
    runs = StudyLog.runs("lesson1-study-alice-two-runs", idea, store=store)
    runs.log("baseline", "baseline", repeats([6, 7, 6]), dataset="dtd")
    runs.log("winner", "final", repeats([7, 8, 7]), dataset="dtd")

    subs, _, _ = study_page.read_store(store, 1)
    [two] = study_page.studies_of(subs)
    assert two.dataset == "dtd" and two.datasets() == ["dtd", "eurosat"]
    tab = study_page.study_view(two)
    assert "<h4>dataset</h4>" not in tab  # no card compares datasets by score
    assert tab.count("<p class='ds'>on dtd</p>") == tab.count(
        "<p class='ds'>on eurosat</p>"
    )
    assert tab.count("<p class='ds'>on eurosat</p>") >= 2  # the lr card and the curves
    for d in ("dtd", "eurosat"):
        best = two.trials_on(d)
        assert set(best[best.winner].dataset) == {d}
    results = study_page.fig_results(two)
    assert "dtd, the winner&#x27;s dataset (the search also drew eurosat)" in results
    space_panel = study_page.fig_space(two)
    assert "class='s on' text-anchor='middle'>dtd</text>" in space_panel


def test_no_runs_stop_the_script_before_page_toml_is_made(tmp_path, folder):
    empty = Store(tmp_path / "nothing")
    with pytest.raises(SystemExit, match="no runs of lesson 1"):
        page(empty, folder)
    assert not (folder / "page.toml").exists()


def test_submitted_runs_read_no_local_page_toml(store, folder, monkeypatch):
    subs, _, _ = study_page.read_store(store, 1)
    monkeypatch.setattr(
        study_page,
        "read_submitted",
        lambda *a: (subs, "alice", "the runs alice submitted"),
    )
    text = page(store, folder, submitted="alice")
    assert not (folder / "page.toml").exists()
    assert text.count(study_page.NOT_SUBMITTED) == len(study_page.OWN)
    assert "Say in three sentences what you did." not in text

    named = folder / "elsewhere.toml"
    named.write_text('message = "Mine, named."\n', encoding="utf-8")
    text = page(store, folder, submitted="alice", page=named)
    assert "Mine, named." in text and study_page.NOT_SUBMITTED not in text
    assert not (folder / "page.toml").exists()


def test_page_names_a_page_toml_for_the_store_too(store, folder):
    named = folder / "draft.toml"
    named.write_text('surprised = "The lr did not matter."\n', encoding="utf-8")
    assert "The lr did not matter." in page(store, folder, page=named)
    assert not (folder / "page.toml").exists()
    with pytest.raises(SystemExit, match="no page.toml at"):
        page(store, folder, page=folder / "missing.toml")


def test_a_store_that_names_no_one_asks_no_one_and_writes_nothing(
    tmp_path, folder, monkeypatch
):
    def refuse(self):
        raise AssertionError("gh was asked")

    monkeypatch.setattr(Store, "login", refuse)
    store = Store(tmp_path / "optuna")
    handed = Idea.handed_out(1)
    search = StudyLog.search("lesson1-notebook", SPACE, handed, store=store)

    def objective(trial: optuna.Trial) -> float:
        search.record(
            trial, RunRecord(role="trial", dataset="dtd", config=SPACE.draw(trial))
        )
        return 0.5

    search.optimize(objective, 2)
    text = page(store, folder)
    assert "<h1>Lesson 1: your study</h1>" in text
    assert not (store.root / "login").exists()


def test_paths_inside_the_repository_are_named_from_its_root(tmp_path):
    inside = study_page.COURSE_ROOT / "artefacts" / "optuna" / "studies.db"
    assert study_page.shown(inside) == "artefacts/optuna/studies.db"
    assert study_page.shown(tmp_path / "studies.db") == str(
        (tmp_path / "studies.db").resolve()
    )


@pytest.mark.parametrize(
    "name", ["../secret.svg", "/etc/hosts", "sub/../../secret.svg"]
)
def test_a_picture_outside_the_folder_is_refused(tmp_path, folder, name):
    (tmp_path / "secret.svg").write_text(
        "<svg><text>secret</text></svg>", encoding="utf-8"
    )
    text = study_page.picture({"picture": name}, folder)
    assert "lies outside the folder of page.toml" in text
    assert "secret</text>" not in text


def test_a_picture_inside_the_folder_is_inlined_as_it_is(folder):
    (folder / "pics").mkdir()
    (folder / "pics" / "idea.svg").write_text(
        "<svg><text>my idea</text></svg>", encoding="utf-8"
    )
    text = study_page.picture({"picture": "pics/idea.svg"}, folder)
    assert "<svg><text>my idea</text></svg>" in text


def test_atlas_code_is_tried_when_the_nightly_fails(folder):
    tried = []

    def run(command, _folder):
        tried.append(command)
        if command == "atlas-code-nightly":
            return None, "atlas-code-nightly took more than 60 seconds"
        return b"<html>the design</html>", ""

    text = study_page.design_tab(folder, which=lambda c: c, run=run)
    assert tried == ["atlas-code-nightly", "atlas-code"]
    assert "atlas-code view" in text and "id='atlas-page'" in text

    def fail(command, _folder):
        return None, f"{command} failed"

    text = study_page.design_tab(folder, which=lambda c: c, run=fail)
    assert "atlas-code-nightly failed; atlas-code failed" in text


def test_a_short_name_drops_what_every_label_shares():
    def named(*labels: str) -> list[str]:
        studies = [
            study_page.Study(
                key=x,
                label=x,
                handed_out=x == "handed out",
                objective="a",
                df=None,
                subs={},
            )
            for x in labels
        ]
        return study_page.short_names(studies)

    assert named("handed out", "bob-wide-head-on-dtd", "bob-deep-skip-on-dtd") == [
        "handed out",
        "wide head",
        "deep skip",
    ]
    assert named("bob-wide-on-dtd") == ["bob wide"]
    assert named("bob-x-on-dtd", "bob-x-on-eurosat") == [
        "bob-x-on-dtd",
        "bob-x-on-eurosat",
    ]


def test_a_title_in_page_toml_names_the_study_everywhere(store, folder):
    page(store, folder)  # makes page.toml, with an empty [titles]
    toml = folder / "page.toml"
    assert "[titles]" in toml.read_text(encoding="utf-8")
    toml.write_text(
        toml.read_text(encoding="utf-8") + '"alice-skip" = "the skip helps"\n',
        encoding="utf-8",
    )
    text = page(store, folder)
    assert "<span class='title'>the skip helps</span>" in text
    assert "title='alice-skip'>the skip helps</a>" in text
    assert "<span class='title'>lr range</span>" in text  # untitled: its short name
    assert study_page.read_titles(folder / "missing.toml") == {}


def test_the_design_tab_opens_on_the_forward_of_the_design(folder):
    assert study_page.design_route(folder) is None
    (folder / "designs.py").write_text(
        "class Head:\n    pass\n\nDESIGN = Design(Head, space=SPACE)\n",
        encoding="utf-8",
    )
    assert study_page.design_route(folder) == "exercises.designs.Head.forward"
    (folder / "designs.py").write_text("DESIGN = (", encoding="utf-8")
    assert study_page.design_route(folder) is None


def test_every_study_tab_has_an_explorer_of_its_finished_trials(store, folder):
    text = page(store, folder)
    blobs = re.findall(r"class='explore-data'>(.*?)</script>", text)
    assert len(blobs) == 2  # one per study, both on dtd
    for blob in blobs:
        data = json.loads(blob)
        names = [f["name"] for f in data["fields"]]
        assert names[0] == "accuracy" and "lr" in names
        assert data["defaults"]["y"] == "accuracy"
        assert all(len(row) == len(names) for row in data["rows"])
    assert "<h2>Explore the trials</h2>" in text


def test_a_view_starred_in_stars_json_is_on_the_page(store, folder):
    star = {"study": "lesson1-study-alice-skip", "dataset": "dtd", "x": "lr",
            "y": "accuracy", "c": ""}  # fmt: skip
    study_page.write_stars(folder / "stars.json", [star])
    text = page(store, folder)
    held = re.search(r"id='stars-data' data-mode='own'[^>]*>(.*?)</script>", text)
    assert held is not None and json.loads(held.group(1)) == [star]


def test_stars_that_are_not_views_are_refused():
    with pytest.raises(TypeError):
        study_page.clean_stars({"x": "lr"})
    with pytest.raises(ValueError, match="names its study"):
        study_page.clean_stars([{"study": "", "x": "lr", "y": "accuracy"}])
    view = {"study": "s", "dataset": "dtd", "x": "lr", "y": "accuracy", "c": "skip"}
    assert study_page.clean_stars([view, dict(view)]) == [view]


def test_the_page_server_keeps_the_stars_in_their_file(tmp_path):
    stars = tmp_path / "stars.json"
    server = study_page.page_server("<p>the page</p>", stars)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{server.server_address[1]}"
    view = {"study": "s", "dataset": "dtd", "x": "FLOPs", "y": "accuracy", "c": ""}

    def post(origin: str | None) -> int:
        headers = {"Content-Type": "application/json"} | (
            {"Origin": origin} if origin else {}
        )
        request = urllib.request.Request(
            address + "/stars", json.dumps([view]).encode(), headers, method="POST"
        )
        try:
            return urllib.request.urlopen(request).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert urllib.request.urlopen(address + "/").read() == b"<p>the page</p>"
        assert post("http://elsewhere.example") == 403 and not stars.exists()
        assert post(address) == 200
        assert json.loads(stars.read_text(encoding="utf-8")) == [view]
        assert json.loads(urllib.request.urlopen(address + "/stars").read()) == [view]
    finally:
        server.shutdown()
        server.server_close()


def test_a_title_is_written_under_titles_and_nothing_else_moves(tmp_path):
    toml = tmp_path / "page.toml"
    toml.write_text(study_page.PAGE_TOML.format(lesson=1), encoding="utf-8")
    before = toml.read_text(encoding="utf-8")
    study_page.write_title(toml, "bob-wide", "a wider head")
    assert study_page.read_titles(toml) == {"bob-wide": "a wider head"}
    study_page.write_title(toml, "bob-wide", 'the "wide" one')
    study_page.write_title(toml, "bob-deep", "deeper")
    assert study_page.read_titles(toml) == {
        "bob-wide": 'the "wide" one',
        "bob-deep": "deeper",
    }
    study_page.write_title(toml, "bob-wide", "")
    study_page.write_title(toml, "bob-deep", "")
    assert toml.read_text(encoding="utf-8") == before
    with pytest.raises(ValueError, match="one line"):
        study_page.write_title(toml, "bob-wide", "two\nlines")


def test_the_page_server_keeps_a_title_in_page_toml(tmp_path):
    toml = tmp_path / "page.toml"
    toml.write_text('message = "mine"\n', encoding="utf-8")
    server = study_page.page_server("<p>the page</p>", tmp_path / "stars.json", toml)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    address = f"http://127.0.0.1:{server.server_address[1]}"

    def post(body: object) -> int:
        request = urllib.request.Request(
            address + "/titles",
            json.dumps(body).encode(),
            {"Content-Type": "application/json", "Origin": address},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request).status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        assert post({"study": "bob-wide", "title": "a wider head"}) == 200
        assert post({"study": "bob-wide", "title": 3}) == 400
        assert json.loads(urllib.request.urlopen(address + "/titles").read()) == {
            "bob-wide": "a wider head"
        }
        assert 'message = "mine"' in toml.read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()
