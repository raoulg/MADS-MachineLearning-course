"""The numbers behind the pictures of class_analysis.py, and the page it writes."""

import math

import class_analysis  # ty: ignore[import-not-found]
import pytest
from conftest import make_submission
from run_submission import idea_id, idea_name  # ty: ignore[import-not-found]


def sub(student, experiment, name, role, value, *, lesson=1, std=None, n=3, **extra):
    """A submission of the shape submit_runs writes, with the attrs the analysis reads."""
    s = make_submission(student, lesson)
    attrs = {"role": role, "dataset": "dtd", "n_repeats": n, **extra}
    if std is not None:
        attrs["accuracy_std"] = std
    run = s.run.model_copy(
        update={
            "experiment": experiment,
            "run_name": name,
            "value": value,
            "attrs": attrs,
            "params": {"lr": 0.01},
        }
    )
    return s.model_copy(update={"run": run})


def test_origin_and_study_come_from_the_experiment_name():
    assert class_analysis.origin_of("lesson1-notebook") == "notebook"
    assert class_analysis.origin_of("lesson1-notebook-runs") == "notebook"
    assert class_analysis.origin_of("lesson1-study-sam-depth-runs") == "study"
    assert class_analysis.origin_of("lesson1-search") == "other"
    assert class_analysis.study_of("lesson1-study-sam-depth-runs") == "sam-depth"
    assert class_analysis.study_of("lesson1-study-sam-depth") == "sam-depth"
    assert class_analysis.study_of("lesson1-notebook") is None


def test_only_the_latest_session_of_a_handed_out_run_counts():
    old = sub(
        "alice",
        "lesson1-notebook-runs",
        "baseline",
        "baseline",
        0.70,
        session="2026-11-01 10:00:00",
    )
    new = sub(
        "alice",
        "lesson1-notebook-runs",
        "baseline",
        "baseline",
        0.80,
        session="2026-11-02 10:00:00",
    )
    other = sub(
        "bob",
        "lesson1-notebook-runs",
        "baseline",
        "baseline",
        0.75,
        session="2026-11-01 10:00:00",
    )
    df = class_analysis.runs_frame([old, new, other])
    assert sorted(df.score) == [0.75, 0.80]


def test_baseline_against_final_in_points_and_sigmas():
    study = "lesson1-study-alice-depth"
    df = class_analysis.runs_frame(
        [
            sub("alice", f"{study}-runs", "baseline", "baseline", 0.80, std=0.01),
            sub("alice", f"{study}-runs", "winner", "final", 0.83, std=0.01),
            sub("alice", study, "trial-00", "trial", 0.84),
            sub("alice", study, "trial-01", "trial", 0.79),
        ]
    )
    [row] = class_analysis.comparisons(df).itertuples()
    assert (row.student, row.study) == ("alice", "alice-depth")
    assert row.gain_pt == pytest.approx(3.0)
    # 0.03 against sqrt(0.01^2/3 + 0.01^2/3) = 0.00816
    assert row.sigma == pytest.approx(0.03 / math.sqrt(2 * 0.01**2 / 3))
    assert row.best_trial == 0.84
    assert row.survived == pytest.approx(0.03 / 0.04)
    assert row.n_trials == 2


def test_a_study_without_a_baseline_or_a_final_is_not_compared():
    study = "lesson1-study-alice-depth"
    df = class_analysis.runs_frame(
        [sub("alice", f"{study}-runs", "baseline", "baseline", 0.80, std=0.01)]
    )
    assert class_analysis.comparisons(df).empty


def test_pruned_trials_are_not_scores_of_the_pooled_search():
    kept = sub("alice", "lesson1-notebook", "trial-00", "trial", 0.8)
    stopped = kept.model_copy(
        update={
            "run": kept.run.model_copy(update={"status": "PRUNED", "run_id": "d" * 32})
        }
    )
    df = class_analysis.runs_frame([kept, stopped])
    assert len(class_analysis.handed_out_trials(df)) == 1


def test_who_built_on_whom():
    bob = sub("bob", "lesson1-study-bob-wide", "wide", "final", 0.70)
    dangling = f"lesson2/carol/{'c' * 32}"
    alice = sub(
        "alice", "lesson1-study-alice-deeper", "deeper", "final", 0.75,
        builds_on=f"{bob.ref().replace('bob', 'bob')},{dangling}",
    )  # fmt: skip
    lk = class_analysis.links(class_analysis.runs_frame([bob, alice]))
    found = lk[lk.found].iloc[0]
    assert (found.student, found.builds_on, found.on_run) == ("alice", "bob", "wide")
    assert found.diff_pt == pytest.approx(5.0)
    lost = lk[~lk.found].iloc[0]
    assert lost.builds_on == "carol" and math.isnan(lost.on_score)


def idea_attrs(author, words, parents=()):
    """The five study attrs of an idea, as optunalog.idea writes them."""
    names = sorted(parents)
    id_ = idea_id(author, words, names)
    return {
        "idea": words,
        "idea_author": author,
        "idea_parents": ",".join(names),
        "idea_id": id_,
        "idea_name": idea_name(id_),
    }


def with_idea(s, attrs):
    """`s` with its study carrying the idea `attrs`."""
    study = s.run.study.model_copy(update={"attrs": {**s.run.study.attrs, **attrs}})
    return s.model_copy(update={"run": s.run.model_copy(update={"study": study})})


ROOT = idea_attrs("course", "a head on frozen features")
GHOST = idea_attrs("carol", "an idea nobody submitted", [ROOT["idea_name"]])
ALICE = idea_attrs("alice", "does depth still help", [ROOT["idea_name"]])
BOB = idea_attrs("bob", "is a wider head better", [ROOT["idea_name"]])
MERGE = idea_attrs("alice", "depth and width together", [ALICE["idea_name"], BOB["idea_name"]])  # fmt: skip
ON_GHOST = idea_attrs("bob", "building on a lost idea", [GHOST["idea_name"]])


def class_with_ideas():
    """Two students on the course's root, a merge of both, and one on a lost parent."""
    study = "lesson1-study-alice-depth"
    return [
        with_idea(sub("alice", "lesson1-notebook-runs", "baseline", "baseline", 0.8), ROOT),
        with_idea(sub("bob", "lesson1-notebook-runs", "baseline", "baseline", 0.8), ROOT),
        with_idea(sub("bob", "lesson1-notebook", "trial-00", "trial", 0.81), ROOT),
        with_idea(sub("alice", f"{study}-runs", "baseline", "baseline", 0.80, std=0.01), ALICE),
        with_idea(sub("alice", f"{study}-runs", "winner", "final", 0.83, std=0.01), ALICE),
        with_idea(sub("alice", study, "trial-00", "trial", 0.84), ALICE),
        with_idea(sub("bob", "lesson1-study-bob-wide", "wide", "final", 0.7), BOB),
        with_idea(sub("alice", "lesson1-study-alice-both", "both", "final", 0.7), MERGE),
        with_idea(sub("bob", "lesson1-study-bob-lost", "lost", "final", 0.7), ON_GHOST),
        sub("carol", "lesson1-notebook-runs", "baseline", "baseline", 0.8),
    ]  # fmt: skip


def test_a_run_without_an_idea_has_empty_idea_columns():
    df = class_analysis.runs_frame(class_with_ideas())
    carol = df[df.student == "carol"].iloc[0]
    assert (carol.idea, carol.idea_name, carol.idea_author, carol.idea_parents) == (
        "", "", "", ""
    )  # fmt: skip
    assert class_analysis.ideas(df[df.student == "carol"], df.iloc[:0]).empty


def test_ideas_merge_share_the_root_and_keep_a_dangling_parent():
    df = class_analysis.runs_frame(class_with_ideas())
    ids = class_analysis.ideas(df, class_analysis.comparisons(df)).set_index("name")
    assert len(ids) == 5  # carol's run carries no idea, her lost idea is no row
    root = ids.loc[ROOT["idea_name"]]
    assert root.author == "course" and root.parents == []
    assert root.students == ["alice", "bob"] and root.n_runs == 3
    merge = ids.loc[MERGE["idea_name"]]
    assert sorted(merge.parents) == sorted([ALICE["idea_name"], BOB["idea_name"]])
    assert merge.missing == [] and merge.verdict == "unmeasured"
    lost = ids.loc[ON_GHOST["idea_name"]]
    assert lost.parents == lost.missing == [GHOST["idea_name"]]
    alice = ids.loc[ALICE["idea_name"]]
    assert alice.gain_pt == pytest.approx(3.0) and alice.verdict == "better"


def test_the_graph_has_a_box_per_idea_and_an_arrow_per_parent_in_layers():
    df = class_analysis.runs_frame(class_with_ideas())
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    fig = class_analysis.fig_ideas(ids)
    boxes = {s.name for s in fig.layout.shapes if s.type == "rect" and s.name}
    boxes = {b for b in boxes if not b.startswith("stripe:")}
    assert boxes == {*ids.name, GHOST["idea_name"]}
    ghost = next(s for s in fig.layout.shapes if s.name == GHOST["idea_name"])
    assert ghost.line.dash == "dash"
    arrows = [a for a in fig.layout.annotations if a.showarrow]
    assert len(arrows) == sum(len(p) for p in ids.parents) == 5
    place = class_analysis.idea_layout(ids)
    for r in ids.itertuples():
        for p in r.parents:
            assert place[p][0] < place[r.name][0]
    assert place[ROOT["idea_name"]][0] == 0
    assert place[MERGE["idea_name"]][0] == 2
    # no two boxes share a place
    assert len(set(place.values())) == len(place)


def test_the_page_says_so_without_any_idea():
    page = class_analysis.build_page(
        class_analysis.runs_frame([sub("carol", "lesson1-notebook-runs", "baseline", "baseline", 0.8)]),
        1,
    )  # fmt: skip
    assert "No submitted run carries an idea yet." in page


def test_the_page_is_one_self_contained_file_with_every_picture():
    from class_demo import demo_submissions  # ty: ignore[import-not-found]

    page = class_analysis.build_page(
        class_analysis.runs_frame(demo_submissions()), 1, "made-up data"
    )
    assert page.count("plotly-graph-div") >= 8
    assert "<script src" not in page and "<link" not in page  # nothing is fetched
    for title in (
        "Who submitted what",
        "The handed-out baselines, side by side",
        "Baseline against final, per study",
        "The handed-out search, pooled over the class",
        "Each student&#x27;s own study",
        "The ideas of the class",
        "Who built on whom",
    ):
        assert title in page
