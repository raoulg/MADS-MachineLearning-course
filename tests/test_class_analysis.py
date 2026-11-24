"""The numbers behind the pictures of class_analysis.py, and the page it writes."""

import math
import re

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
    # a stored std is over n=3 repeats with ddof=0: the standard error is
    # sqrt(0.01^2/2 + 0.01^2/2) = 0.01
    assert row.sigma == pytest.approx(0.03 / math.sqrt(2 * 0.01**2 / 2))
    assert row.best_trial == 0.84
    assert row.survived == pytest.approx(0.03 / 0.04)
    assert row.n_trials == 2


def test_the_page_and_compare_give_the_same_sigma_for_the_same_repeats():
    import numpy as np

    from hpo.measure.compare import Comparison

    base, final = np.array([0.80, 0.81, 0.795]), np.array([0.82, 0.835, 0.81, 0.83])
    study = "lesson1-study-alice-depth"
    df = class_analysis.runs_frame(
        [
            sub("alice", f"{study}-runs", "baseline", "baseline", float(base.mean()),
                std=float(base.std()), n=len(base)),
            sub("alice", f"{study}-runs", "winner", "final", float(final.mean()),
                std=float(final.std()), n=len(final)),
        ]
    )  # fmt: skip
    [row] = class_analysis.comparisons(df).itertuples()
    assert row.sigma == pytest.approx(Comparison.of("final", final, base).sigmas)


def test_a_single_run_has_no_sigma_and_its_card_names_no_seeds():
    study = "lesson1-study-alice-depth"
    df = class_analysis.runs_frame(
        [
            sub("alice", f"{study}-runs", "baseline", "baseline", 0.80, std=0.0, n=1),
            sub("alice", f"{study}-runs", "winner", "final", 0.83, std=0.0, n=1),
        ]
    )
    [row] = class_analysis.comparisons(df).itertuples()
    assert math.isnan(row.sigma)
    r = class_analysis.SimpleNamespace(
        name="x", final=0.83, final_std=0.0, final_n=1, dataset="dtd"
    )
    lines = class_analysis.found_line(
        r, None, {}, class_analysis.edges(df.iloc[:0].assign(name=[], parents=[]))
    )
    assert lines == [("final", "<b class='num'>0.830</b>")]


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


def idea_attrs(author, words, parents=()):
    """The five study attrs of an idea, as hpo.record.ideas.Idea.attrs gives them."""
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


def test_the_lanes_hold_a_card_per_students_idea_and_link_only_submitted_ideas():
    df = class_analysis.runs_frame(class_with_ideas())
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    page = class_analysis.lanes(ids, class_analysis.edges(ids), None, {})
    cards = re.findall(r"data-name='([^']+)' data-parents='([^']*)'", page)
    names = {n for n, _ in cards}
    assert names == set(ids.name) - {
        ROOT["idea_name"]
    }  # the handed-out idea is no card
    parents = {
        n: class_analysis.json.loads(class_analysis.html.unescape(p)) for n, p in cards
    }
    assert sorted(parents[MERGE["idea_name"]]) == sorted(
        [ALICE["idea_name"], BOB["idea_name"]]
    )
    assert parents[ALICE["idea_name"]] == []  # on the handed-out idea: no link
    assert parents[ON_GHOST["idea_name"]] == []
    assert f"on {GHOST['idea_name']}, not submitted" in page
    lanes = re.findall(r"<div class='lane-label'><b>([^<]+)</b>", page)
    assert sorted(lanes) == ["alice", "bob"]


def test_the_page_says_so_without_any_idea():
    page = class_analysis.build_page(
        class_analysis.runs_frame([sub("carol", "lesson1-notebook-runs", "baseline", "baseline", 0.8)]),
        1,
    )  # fmt: skip
    assert "No submitted run carries an idea of a student yet." in page
    assert "id='drawer-data'" not in page


def test_the_page_is_one_self_contained_file_with_every_picture():
    from class_demo import demo_submissions  # ty: ignore[import-not-found]

    page = class_analysis.build_page(
        class_analysis.runs_frame(demo_submissions()), 1, "made-up data"
    )
    assert page.count("plotly-graph-div") >= 3
    assert "<script src" not in page and "<link" not in page  # nothing is fetched
    for tab in (
        "investigated",
        "built",
        "noise",
        "searched",
        "interactions",
        "cost",
        "students",
    ):
        assert f"id='panel-{tab}'" in page and f"id='tab-{tab}'" in page
    # the view of the investigated goals opens first; the others wait behind their tab
    assert "id='panel-investigated' aria-labelledby='tab-investigated'>" in page
    assert "id='panel-built' aria-labelledby='tab-built' hidden>" in page
    for goal in ("1.5", "1.6", "1.9", "1.10"):
        assert f"<span class='goal num'>{goal}</span>" in page
    assert "ablation" not in page.lower().replace("ablation_", "")
    assert "σ" not in page


def test_an_attr_of_another_type_is_read_as_absent():
    odd = sub(
        "mallory", "lesson1-notebook-runs", "baseline", {"a": 1}, 0.8,
        session=7, config=["not", "a", "dict"], architecture=3, builds_on=[1],
    )  # fmt: skip
    odd.run.attrs["dataset"] = 12
    df = class_analysis.runs_frame(
        [odd, sub("carol", "lesson1-notebook-runs", "baseline", "baseline", 0.8)]
    )
    [row] = df[df.student == "mallory"].itertuples()
    assert (row.role, row.dataset, row.builds_on) == ("-", "unknown", "")
    assert row.session is None and row.architecture is None
    assert "id='panel-students'" in class_analysis.build_page(df, 1)


def test_a_students_words_and_names_hold_no_tag():
    words = "<a href='https://example.org'>click</a>"
    a = idea_attrs("a", words, [ROOT["idea_name"]])
    df = class_analysis.runs_frame(study_runs("a", "a-q", a, 0.7, 0.8))
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    page = class_analysis.lanes(ids, class_analysis.edges(ids), None, {})
    assert words not in page and "&lt;a href=" in page


@pytest.mark.parametrize(
    ("sigma", "word"),
    [(2.0, "better"), (1.9, "inside"), (-1.9, "inside"), (-2.0, "worse"),
     (None, "unmeasured"), (math.nan, "unmeasured")],
)  # fmt: skip
def test_a_verdict_is_two_sigmas_either_way(sigma, word):
    assert class_analysis.verdict(sigma) == word


def test_a_parent_that_is_not_submitted_is_named_on_the_card():
    df = class_analysis.runs_frame(
        [with_idea(sub("bob", "lesson1-study-bob-x", "trial-00", "trial", 0.8),
                   idea_attrs("bob", "an idea on a parent nobody sent", ["plaid-eagle-75"]))]
    )  # fmt: skip
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    [name] = ids.name
    depth = class_analysis.idea_depths(ids)
    assert depth["plaid-eagle-75"] == 0 and depth[name] == 1
    page = class_analysis.lanes(ids, class_analysis.edges(ids), None, {})
    assert "on plaid-eagle-75, not submitted" in page


def test_an_idea_searched_twice_shows_the_study_with_the_most_trials():
    attrs = idea_attrs("alice", "depth on dtd once more")
    runs = []
    for study, final, n in (("alice-aaa", 0.80, 1), ("alice-zzz", 0.85, 3)):
        full = f"lesson1-study-{study}"
        runs += [
            sub("alice", f"{full}-runs", "baseline", "baseline", 0.80, std=0.01),
            sub("alice", f"{full}-runs", "winner", "final", final, std=0.01),
            *(sub("alice", full, f"trial-{i:02d}", "trial", 0.8) for i in range(n)),
        ]
    df = class_analysis.runs_frame([with_idea(r, attrs) for r in runs])
    [row] = class_analysis.ideas(df, class_analysis.comparisons(df)).itertuples()
    assert (row.study, row.verdict) == ("alice-zzz", "better")


def test_the_page_reads_any_logged_metric_and_names_it():
    import class_demo  # ty: ignore[import-not-found]

    subs = class_demo.demo_submissions()
    names = class_analysis.metric_names(subs)
    assert names[:2] == ["accuracy", "macro_recall"]
    assert "recall/blotchy" in names
    by_accuracy = class_analysis.runs_frame(subs, "accuracy")
    by_recall = class_analysis.runs_frame(subs, "macro_recall")
    assert (by_recall.score < by_accuracy.score).all()
    page = class_analysis.build_page(by_recall, 1, "", "macro_recall", names)
    assert "macro_recall of the handed-out baseline" in page
    assert "Every picture reads <b>macro_recall</b>" in page


def test_the_page_reads_the_objective_most_studies_search():
    s = sub("alice", "lesson2-study-a-runs", "final", "final", 0.5)
    on_recall = s.model_copy(
        update={
            "run": s.run.model_copy(
                update={
                    "study": s.run.study.model_copy(
                        update={"attrs": {"objective": "macro_recall"}}
                    )
                }
            )
        }
    )
    assert class_analysis.page_metric([on_recall, on_recall, s]) == "macro_recall"
    assert class_analysis.page_metric([]) == "accuracy"


def test_a_run_without_the_metric_has_no_score():
    old = sub("alice", "lesson1-study-a-runs", "final", "final", 0.8)
    df = class_analysis.runs_frame([old], "macro_recall")
    assert df.score.isna().all()


def test_a_parent_of_another_lesson_is_named_with_its_lesson():
    s = with_idea(
        sub("bram", "lesson2-study-b-runs", "final", "final", 0.5, lesson=2),
        idea_attrs("bram", "a gate on beats", ["blissful-bee-18"]),
    )
    df = class_analysis.runs_frame([s])
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    page = class_analysis.lanes(
        ids, class_analysis.edges(ids), None, {"blissful-bee-18": 1}
    )
    assert "on blissful-bee-18 (lesson 1)" in page


# ---- the ideas on top of the page, and the window ------------------------------------


def study_runs(student, name, attrs, base, final, std=0.01, trials=(), **study):
    """The runs of one student's study as `hpo.exercise.study.Exercise` logs them: the
    baseline and the winner with their spread, and `trials` as (score, status, params),
    each with its curve; `study` adds study attrs (expects, searched_axes, objective)."""
    full = f"lesson1-study-{name}"
    attrs = {**attrs, **study}
    runs = [
        sub(student, f"{full}-runs", "baseline", "baseline", base, std=std),
        sub(student, f"{full}-runs", "winner", "final", final, std=std),
    ]
    for i, (score, status, params) in enumerate(trials):
        t = sub(
            student,
            full,
            f"trial-{i:02d}",
            "trial",
            score,
            accuracy=[score - 0.1, score],
        )
        run = t.run.model_copy(
            update={"status": status, "params": params, "start_time": i}
        )
        runs.append(t.model_copy(update={"run": run}))
    return [with_idea(r, attrs) for r in runs]


def with_attrs(s, **attrs):
    """`s` with `attrs` added to its run's user attrs."""
    run = s.run.model_copy(update={"attrs": {**s.run.attrs, **attrs}})
    return s.model_copy(update={"run": run})


def rehearsal():
    """The plan of the rehearsal: a1 and b2 on the course's idea, b1 on a1, c1 on b2, c2 a
    merge of b1 and c1, d1 on c1."""
    a1 = idea_attrs("a", "does a wider head still help", [ROOT["idea_name"]])
    b2 = idea_attrs("b", "skip or gate, which one carries it", [ROOT["idea_name"]])
    b1 = idea_attrs("b", "a second layer on the wide head", [a1["idea_name"]])
    c1 = idea_attrs("c", "the skip with a lower learning rate", [b2["idea_name"]])
    c2 = idea_attrs(
        "c", "wide and deep with the skip", [b1["idea_name"], c1["idea_name"]]
    )
    d1 = idea_attrs("d", "input noise on the skip head", [c1["idea_name"]])
    runs = [
        with_idea(
            sub("a", "lesson1-notebook-runs", "baseline", "baseline", 0.70, std=0.01),
            ROOT,
        )
    ]
    for student, name, attrs, final in (
        ("a", "a-wide", a1, 0.75), ("b", "b-skip", b2, 0.74), ("b", "b-deep", b1, 0.755),
        ("c", "c-lr", c1, 0.76), ("c", "c-both", c2, 0.80), ("d", "d-noise", d1, 0.70),
    ):  # fmt: skip
        runs += study_runs(student, name, attrs, 0.70, final, expects=f"{name} pays")
    return runs, {"a1": a1, "b2": b2, "b1": b1, "c1": c1, "c2": c2, "d1": d1}


def test_an_idea_stands_one_column_right_of_its_furthest_parent():
    runs, named = rehearsal()
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df)).set_index("name")
    rounds = {k: ids.loc[v["idea_name"], "round"] for k, v in named.items()}
    assert rounds == {"a1": 1, "b2": 1, "b1": 2, "c1": 2, "c2": 3, "d1": 3}
    assert ids.loc[ROOT["idea_name"], "round"] == 0
    assert ids.loc[named["c2"]["idea_name"], "expects"] == "c-both pays"
    eds = class_analysis.edges(ids.reset_index())
    assert len(eds) == 7 and eds.number.sum() == 5  # the two from the course carry none


def test_a_link_carries_the_childs_final_on_its_parents_final_and_says_whether_it_counts():
    runs, named = rehearsal()
    df = class_analysis.runs_frame(runs)
    assert (
        class_analysis.class_noise(df) is None
    )  # every baseline scores 0.70: no spread
    eds = class_analysis.edges(class_analysis.ideas(df, class_analysis.comparisons(df)))
    arrow = eds.set_index(["parent", "child"])
    # two finals of three repeats, each a std of 0.01 (ddof=0)
    se = math.sqrt(2 * 0.01**2 / 2)

    def at(parent, child):
        return arrow.loc[(named[parent]["idea_name"], named[child]["idea_name"])]

    inside = at("a1", "b1")  # 0.755 on 0.75
    assert inside.diff_pt == pytest.approx(0.5) and inside.verdict == "inside"
    assert inside.sigma == pytest.approx(0.005 / se)
    assert inside.band_pt == pytest.approx(200 * se)
    assert at("c1", "c2").verdict == "better"  # 0.80 on 0.76
    worse = at("c1", "d1")  # 0.70 on 0.76
    assert worse.diff_pt == pytest.approx(-6.0) and worse.verdict == "worse"
    course = arrow.loc[(ROOT["idea_name"], named["a1"]["idea_name"])]
    assert not course.number and "handed-out" in course.why


def test_a_link_reads_the_class_noise_when_the_baseline_was_rerun():
    a = idea_attrs("a", "the parent", [ROOT["idea_name"]])
    b = idea_attrs("b", "the child", [a["idea_name"]])
    runs = study_runs("a", "a-p", a, 0.70, 0.75) + study_runs("b", "b-c", b, 0.72, 0.76)
    runs += [sub(f"s{i}", "lesson1-notebook-runs", "baseline", "baseline", 0.71, std=0.01)
             for i in range(2)]  # fmt: skip
    df = class_analysis.runs_frame(runs)
    noise = class_analysis.class_noise(df)
    assert noise is not None and noise.n == 4
    assert noise.spread == pytest.approx(
        class_analysis.pd.Series([0.70, 0.72, 0.71, 0.71]).std()
    )
    assert noise.band == pytest.approx(2 * math.sqrt(2) * noise.spread)
    eds = class_analysis.edges(
        class_analysis.ideas(df, class_analysis.comparisons(df)), noise
    )
    [arrow] = eds[eds.number].itertuples()
    assert arrow.diff_pt == pytest.approx(1.0)
    assert arrow.band_pt == pytest.approx(100 * noise.band)


GHOST_CHILD = idea_attrs("dave", "on a parent nobody sent", ["plaid-eagle-75"])


def test_a_link_between_datasets_or_from_a_lost_parent_has_no_number():
    alice = idea_attrs("alice", "does depth still help", [ROOT["idea_name"]])
    on_beats = idea_attrs("bob", "depth on another dataset", [alice["idea_name"]])
    on_recall = idea_attrs("carol", "depth searched on recall", [alice["idea_name"]])
    runs = study_runs("alice", "alice-a", alice, 0.7, 0.8)
    runs += [
        with_attrs(r, dataset="beats")
        for r in study_runs("bob", "bob-b", on_beats, 0.7, 0.8)
    ]
    runs += study_runs(
        "carol", "carol-c", on_recall, 0.7, 0.8, objective="macro_recall"
    )
    runs = [
        with_attrs(r, metrics={"accuracy": r.run.value, "macro_recall": r.run.value})
        for r in runs
    ]
    runs += study_runs("dave", "dave-d", GHOST_CHILD, 0.7, 0.8)
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    eds = class_analysis.edges(ids)
    why = dict(zip(eds.child, eds.why, strict=True))
    numbered = dict(zip(eds.child, eds.number, strict=True))
    assert why[on_beats["idea_name"]].startswith("different datasets")
    assert not numbered[on_beats["idea_name"]]
    # a search on another objective is still read on the page's metric
    assert numbered[on_recall["idea_name"]]
    assert "not among the submissions" in why[GHOST_CHILD["idea_name"]]
    page = class_analysis.lanes(ids, eds, None, {})
    assert "searched on macro_recall" in page


def test_the_drawer_shows_an_idea_its_words_and_its_links():
    runs, named = rehearsal()
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    eds = class_analysis.edges(ids)
    data = class_analysis.json.loads(
        class_analysis.drawer_data(df, ids, eds, None, "accuracy")
    )
    assert set(data) == {v["idea_name"] for v in named.values()}
    c2 = data[named["c2"]["idea_name"]]
    assert "c-both pays" in c2["head"]
    assert (
        named["b1"]["idea_name"] in c2["head"]
        and named["c1"]["idea_name"] in c2["head"]
    )
    assert set(c2["figs"]) == {"trials", "lines", "curves"}
    c1 = data[named["c1"]["idea_name"]]
    assert "Built on by" in c1["head"] and named["d1"]["idea_name"] in c1["head"]


def test_the_drawer_draws_the_trials_of_the_study_with_the_best_finished_one_as_winner():
    alice = idea_attrs("alice", "does depth still help", [ROOT["idea_name"]])
    runs = study_runs(
        "alice", "alice-a", alice, 0.7, 0.8, searched_axes="hidden_dim,lr",
        trials=[(0.78, "COMPLETE", {"lr": 1e-3, "hidden_dim": 64}),
                (0.90, "PRUNED", {"lr": 1e-2, "hidden_dim": 128}),
                (0.81, "COMPLETE", {"lr": 1e-4, "hidden_dim": 1024})],
    )  # fmt: skip
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    trials = class_analysis.idea_trials(df, ids, alice["idea_name"])
    assert list(trials.score[trials.winner]) == [0.81]  # a pruned trial never wins
    assert trials.curve.iloc[0] == pytest.approx([0.68, 0.78])
    fig, below = class_analysis.fig_lines(
        trials, ["hidden_dim", "lr"], None, "accuracy"
    )
    [par] = fig.data
    assert below == 0 and len(par.dimensions[0].values) == 2  # the finished trials
    assert [d.label for d in par.dimensions] == ["hidden width", "lr", "accuracy"]


def test_the_summary_counts_ideas_and_building_on_a_classmate():
    runs, _ = rehearsal()
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    text = class_analysis.class_summary(df, ids, class_analysis.edges(ids))
    assert "<b>4 students</b>" in text and "<b>6 ideas</b>" in text
    # b1 on a1, c1 on b2, c2 on b1, d1 on c1; c2 on c's own c1 is no classmate's
    assert "<b>4</b> build on a classmate’s idea" in text


def test_a_class_with_only_studies_of_its_own_says_what_is_missing():
    alice = idea_attrs("alice", "does depth still help", [ROOT["idea_name"]])
    df = class_analysis.runs_frame(study_runs("alice", "alice-a", alice, 0.7, 0.8))
    page = class_analysis.build_page(df, 1)
    assert "Fewer than three runs of the handed-out baseline were submitted." in page
    assert "id='drawer-data'" in page and "No finished trial submitted." in page


def test_the_demo_keeps_a_merge_a_missing_parent_and_curves():
    from class_demo import demo_submissions  # ty: ignore[import-not-found]

    df = class_analysis.runs_frame(demo_submissions())
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    assert (ids.parents.map(len) == 2).any()
    assert ids.missing.map(bool).any()
    assert (ids[ids.author != "course"].expects != "").all()
    assert df[df.role == "trial"].curve.map(bool).all()
    assert class_analysis.edges(ids).number.any()


# ---- a large class, the teacher, and what an arrow or a window shows -----------------


def big_class(n=30):
    """`n` students, each with a baseline and a final of a study of their own on the
    course's root."""
    runs = []
    for i in range(n):
        student = f"s{i:02d}"
        idea = idea_attrs(student, f"idea number {i}", [ROOT["idea_name"]])
        runs += study_runs(student, f"{student}-x", idea, 0.70, 0.70 + i / 1000)
    return runs


def test_the_page_builds_for_thirty_students_with_a_baseline_and_a_final_each():
    df = class_analysis.runs_frame(big_class())
    page = class_analysis.build_page(df, 1)
    assert "id='drawer-data'" in page
    assert page.count("<div class='lane-label'><b>") == 30


def test_a_lane_is_one_student_and_lanes_stand_in_the_order_their_first_idea_ran():
    runs, _ = rehearsal()
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    page = class_analysis.lanes(ids, class_analysis.edges(ids), None, {})
    lanes = re.findall(
        r"<div class='lane-label'><b>([^<]+)</b><span>([^<]+)</span>", page
    )
    assert dict(lanes) == {"a": "1 idea", "b": "2 ideas", "c": "2 ideas", "d": "1 idea"}
    first = df[df.origin == "study"].groupby("student").start.min().sort_values()
    assert [name for name, _ in lanes] == list(first.index)


def test_a_merge_is_as_deep_as_its_deepest_parent_not_its_nearest():
    runs, named = rehearsal()
    # on a1 (depth 1) and b1 (depth 2): depth 3 by the longest path, 2 by the shortest
    e1 = idea_attrs("e", "the wide head and its second layer", [named["a1"]["idea_name"], named["b1"]["idea_name"]])  # fmt: skip
    runs += study_runs("e", "e-merge", e1, 0.70, 0.76)
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df)).set_index("name")
    assert ids.loc[e1["idea_name"], "round"] == 3


def test_a_link_at_one_and_a_half_standard_errors_is_within_the_noise():
    a = idea_attrs("a", "the parent", [ROOT["idea_name"]])
    b = idea_attrs("b", "the child", [a["idea_name"]])
    # two finals of three repeats, std 0.01 each: the standard error is 0.01
    child = 0.75 + 1.5 * 0.01
    runs = study_runs("a", "a-p", a, 0.70, 0.75) + study_runs(
        "b", "b-c", b, 0.70, child
    )
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    eds = class_analysis.edges(ids)
    [arrow] = eds[eds.number].itertuples()
    assert arrow.sigma == pytest.approx(1.5)
    assert arrow.verdict == "inside"
    by = {r.name: r for r in class_analysis.rows(ids)}
    lines = class_analysis.found_line(by[b["idea_name"]], None, by, eds)
    assert (
        "inside",
        "<span class='num'>+1.5 pt</span> on a’s "
        + a["idea_name"]
        + " · within the noise",
    ) in lines


def test_each_idea_has_a_drawer_of_its_own():
    runs, named = rehearsal()
    df = class_analysis.runs_frame(
        runs
        + study_runs(
            "a", "a-wide", named["a1"], 0.70, 0.75, searched_axes="lr",
            trials=[(0.74, "COMPLETE", {"lr": 1e-3}), (0.75, "COMPLETE", {"lr": 1e-2})],
        )
        + study_runs(
            "c", "c-lr", named["c1"], 0.70, 0.76, searched_axes="lr",
            trials=[(0.72, "COMPLETE", {"lr": 1e-4}), (0.76, "COMPLETE", {"lr": 1e-3})],
        )
    )  # fmt: skip
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    data = class_analysis.json.loads(
        class_analysis.drawer_data(df, ids, class_analysis.edges(ids), None, "accuracy")
    )
    a1, c1 = data[named["a1"]["idea_name"]], data[named["c1"]["idea_name"]]
    for key in ("trials", "lines", "curves"):
        assert a1["figs"][key] != c1["figs"][key], key


def test_an_idea_of_two_studies_says_so_and_the_drawer_lists_both():
    a = idea_attrs("a", "one idea, two tries", [ROOT["idea_name"]])
    runs = study_runs("a", "a-big", a, 0.7, 0.8, trials=[(0.8, "COMPLETE", {})] * 3)
    runs += study_runs("a", "a-small", a, 0.7, 0.75, trials=[(0.75, "COMPLETE", {})])
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    [r] = [r for r in class_analysis.rows(ids) if r.author == "a"]
    assert [s[1] for s in r.studies] == ["a-big", "a-small"]
    data = class_analysis.json.loads(
        class_analysis.drawer_data(df, ids, class_analysis.edges(ids), None, "accuracy")
    )
    head = data[r.name]["head"]
    assert (
        "2 studies of this idea: shown a-big (3 trials); also a-small (1 trial)" in head
    )


def test_a_quote_in_a_students_words_shows_as_a_quote():
    a = idea_attrs("a", 'the "wide" head', [ROOT["idea_name"]])
    df = class_analysis.runs_frame(study_runs("a", "a-q", a, 0.7, 0.8))
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    page = class_analysis.lanes(ids, class_analysis.edges(ids), None, {})
    assert class_analysis.html.escape('the "wide" head') in page


def test_a_dataset_name_in_a_reason_is_escaped():
    a = idea_attrs("a", "the parent", [ROOT["idea_name"]])
    b = idea_attrs("b", "the child", [a["idea_name"]])
    runs = study_runs("a", "a-p", a, 0.7, 0.75) + [
        with_attrs(r, dataset="<b>beats</b>")
        for r in study_runs("b", "b-c", b, 0.7, 0.76)
    ]
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    eds = class_analysis.edges(ids)
    drawer = class_analysis.drawer_data(df, ids, eds, None, "accuracy")
    assert "<b>beats</b>" not in drawer and "&lt;b&gt;beats" in drawer


def test_an_idea_without_trials_says_so():
    fig, below = class_analysis.fig_lines(class_analysis.pd.DataFrame(
        {"status": [], "score": []}), [], None, "accuracy")  # fmt: skip
    assert below == 0
    assert fig.layout.annotations[0].text == "no finished trial above the baseline"


def teacher_class():
    """Four students' studies, and the teacher's handed-out notebook run twice: an
    earlier execution and the latest, which holds two baselines."""
    runs, _ = rehearsal()
    for session, n in (("2026-11-16 22:00:00", 1), ("2026-11-16 23:00:00", 2)):
        for i in range(n):
            runs.append(
                with_idea(
                    sub("tutor", "lesson1-notebook-runs", "baseline", "baseline", 0.71 + i / 100,
                        std=0.01, session=session),
                    ROOT,
                )
            )  # fmt: skip
    return runs


def test_a_class_with_a_teacher_counts_its_students_without_the_teacher():
    runs = teacher_class()
    df = class_analysis.runs_frame(runs)
    page = class_analysis.build_page(df, 1, "owner/runs at main", teacher="tutor",
                                     submitted=len(runs))  # fmt: skip
    assert "<b>4 students</b>" in page
    shown = class_analysis.as_teacher(df, "tutor")
    assert class_analysis.students(shown) == ["a", "b", "c", "d"]
    base = class_analysis.handed_out_baselines(shown)
    assert list(base.student).count("tutor · teacher") == 1  # one baseline per person
    ids = class_analysis.ideas(shown, class_analysis.comparisons(shown))
    course = ids[ids.author == "course"]
    assert len(course) == 1 and "tutor · teacher" in course.iloc[0].students


def test_an_idea_of_the_teacher_stands_in_the_teachers_lane():
    t = idea_attrs("tutor", "the teacher's own try", [ROOT["idea_name"]])
    a = idea_attrs("a", "a student's try", [ROOT["idea_name"]])
    df = class_analysis.runs_frame(
        study_runs("tutor", "tutor-t", t, 0.7, 0.8)
        + study_runs("a", "a-a", a, 0.7, 0.8)
    )
    shown = class_analysis.as_teacher(df, "tutor")
    ids = class_analysis.ideas(shown, class_analysis.comparisons(shown))
    page = class_analysis.lanes(ids, class_analysis.edges(ids), None, {})
    assert "<b>tutor · teacher</b>" in page


def test_the_page_says_one_number_of_runs_and_names_the_rest():
    runs = teacher_class()
    df = class_analysis.runs_frame(runs)
    assert len(runs) - len(df) == 1  # the teacher's earlier execution
    page = class_analysis.build_page(df, 1, "owner/runs at main", teacher="tutor",
                                     submitted=len(runs))  # fmt: skip
    line = class_analysis.source_line("owner/runs at main", len(runs), len(df))
    assert line.startswith(
        f"{len(runs)} runs submitted to owner/runs at main. 1 of them"
    )
    assert f"the page reads {len(df)} runs." in line
    assert class_analysis.html.escape(line) in page
    same = class_analysis.source_line("owner/runs at main", len(df), len(df))
    assert same == f"{len(df)} runs submitted to owner/runs at main."


def test_the_class_repository_names_its_owner():
    m = class_analysis.GITHUB.search("https://github.com/someone/class-runs.git")
    assert m is not None and m.groups() == ("someone", "class-runs")
    m = class_analysis.GITHUB.search("git@github.com:owner/runs")
    assert m is not None and m.groups() == ("owner", "runs")


def test_the_demo_has_a_teacher_who_ran_the_notebook_twice():
    from class_demo import DEMO_TEACHER, demo_submissions  # ty: ignore[import-not-found]

    subs = demo_submissions()
    df = class_analysis.runs_frame(subs)
    assert len(subs) - len(df) == 25  # the teacher's earlier baseline and 24 trials
    page = class_analysis.build_page(df, 1, "made-up data", teacher=DEMO_TEACHER,
                                     submitted=len(subs))  # fmt: skip
    assert "<b>8 students</b>" in page


def test_a_link_compares_final_with_final_not_gain_with_gain():
    a = idea_attrs("a", "the parent", [ROOT["idea_name"]])
    b = idea_attrs("b", "the child", [a["idea_name"]])
    # the same gain of 5 pt, on baselines 1 pt apart: the child's final is 1 pt higher
    runs = study_runs("a", "a-p", a, 0.70, 0.75) + study_runs("b", "b-c", b, 0.71, 0.76)
    df = class_analysis.runs_frame(runs)
    eds = class_analysis.edges(class_analysis.ideas(df, class_analysis.comparisons(df)))
    [arrow] = eds[eds.number].itertuples()
    assert arrow.diff_pt == pytest.approx(1.0)


def test_a_link_without_a_final_or_with_a_single_run_has_no_number():
    a = idea_attrs("a", "the parent", [ROOT["idea_name"]])
    b = idea_attrs("b", "no final", [a["idea_name"]])
    c = idea_attrs("c", "one run", [a["idea_name"]])
    runs = study_runs("a", "a-p", a, 0.70, 0.75)
    runs += [
        r
        for r in study_runs("b", "b-b", b, 0.70, 0.76)
        if r.run.attrs["role"] != "final"
    ]
    runs += [
        with_attrs(r, n_repeats=1) if r.run.attrs["role"] == "final" else r
        for r in study_runs("c", "c-c", c, 0.70, 0.76)
    ]
    df = class_analysis.runs_frame(runs)
    eds = class_analysis.edges(class_analysis.ideas(df, class_analysis.comparisons(df)))
    why = dict(zip(eds.child, eds.why, strict=True))
    assert not eds[eds.parent == a["idea_name"]].number.any()
    assert why[b["idea_name"]] == "no final at both ends"
    assert why[c["idea_name"]] == "a side has a single run: no standard error"


def test_the_page_can_load_plotly_from_cdnjs_with_its_hash():
    from plotly.offline import get_plotlyjs_version

    df = class_analysis.runs_frame(teacher_class())
    small = class_analysis.build_page(df, 1, plotly="cdn")
    url = f"cdnjs.cloudflare.com/ajax/libs/plotly.js/{get_plotlyjs_version()}/plotly.min.js"
    assert url in small and 'integrity="sha384-' in small
    assert len(small) < len(class_analysis.build_page(df, 1)) - 4_000_000


def test_a_page_links_the_course_start_page_only_when_asked():
    df = class_analysis.runs_frame(
        [sub("carol", "lesson1-notebook-runs", "baseline", "baseline", 0.8)]
    )
    own = class_analysis.build_page(df, 1)
    assert "course-link" not in own and "MADS-ML start" not in own
    start = "https://raoulg.github.io/MADS-MachineLearning-course/index.html"
    published = class_analysis.build_page(df, 1, course=start)
    assert f"<a href='{start}'>← MADS-ML start</a>" in published
    assert published.index("course-link'") < published.index("<h1>")
    ids = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    rest = published.replace(class_analysis.course_link(start), "")
    assert ids.sub("id", rest) == ids.sub("id", own)  # plotly draws fresh div ids


def test_the_command_line_links_the_start_page_only_with_course_link(
    tmp_path, monkeypatch
):
    start = "https://example.invalid/course/index.html"
    pages = {}
    for name, extra in (("linked", ["--course-link", start]), ("own", [])):
        out = tmp_path / f"{name}.html"
        argv = ["class_analysis.py", "--demo", "--plotly", "cdn", "--out", str(out)]
        monkeypatch.setattr("sys.argv", argv + extra)
        class_analysis.main()
        pages[name] = out.read_text(encoding="utf-8")
    assert pages["linked"].count(class_analysis.course_link(start)) == 1
    assert "course-link" not in pages["own"]


def test_an_idea_stands_under_the_goals_whose_axes_it_searched():
    wide = idea_attrs("a", "does a wider head still help", [ROOT["idea_name"]])
    noisy = idea_attrs("b", "input noise against overfitting", [ROOT["idea_name"]])
    runs = [
        with_idea(
            sub("a", "lesson1-notebook-runs", "baseline", "baseline", 0.70, std=0.01),
            ROOT,
        )
    ]
    runs += study_runs("a", "a-wide", wide, 0.70, 0.75, searched_axes="hidden_dim,lr")
    runs += study_runs("b", "b-noise", noisy, 0.70, 0.71, searched_axes="dropout")
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    grouped = class_analysis.ideas_by_theme(df, ids, class_analysis.themes_of(1, ids))
    names = {k: [r.name for r in v] for k, v in grouped.items()}
    assert names["capacity"] == [wide["idea_name"]]  # hidden_dim; lr makes no theme
    assert names["overfitting"] == [noisy["idea_name"]]
    assert names["encoder"] == [] and names["other"] == []


def test_a_lesson_without_goals_gets_a_theme_per_searched_axis():
    gated = idea_attrs("a", "does a skip help", [ROOT["idea_name"]])
    df = class_analysis.runs_frame(
        study_runs("a", "a-skip", gated, 0.70, 0.72, searched_axes="skip,lr")
    )
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    assert [t.key for t in class_analysis.themes_of(9, ids)] == ["skip"]


def test_training_compute_counts_parameters_rows_and_epochs():
    trials = class_analysis.pd.DataFrame({
        "n_params": [1000, 1000, None], "epochs_run": [15, 15, 15], "dataset": ["dtd", "dtd", "dtd"],
        "p_n_train": [None, 100, None],
    })  # fmt: skip
    flops = class_analysis.train_flops(trials)
    assert flops[0] == 6 * 1000 * 3760 * 15  # the whole train split of dtd
    assert flops[1] == 6 * 1000 * 100 * 15  # a subset of 100 rows
    assert class_analysis.math.isnan(flops[2])
    assert class_analysis.flops_text(2.5e9) == "2.5 G"


def test_the_lines_carry_both_colourings_with_the_midpoint_at_the_noise_of_the_best():
    noise = class_analysis.Noise(center=0.70, spread=0.005, n=10, dataset="dtd")
    trials = class_analysis.pd.DataFrame({
        "status": ["COMPLETE"] * 4, "score": [0.72, 0.74, 0.76, 0.69],
        "p_hidden_dim": [64, 128, 256, 512], "p_encoder": ["dinov2-small", "dinov2-small", "dinov2-large", "dinov2-small"],
    })  # fmt: skip
    fig, below = class_analysis.fig_lines(
        trials, ["hidden_dim", "encoder"], noise, "accuracy"
    )
    assert below == 1
    meta = fig.layout.meta
    assert [e for e, _ in meta["encoders"]] == ["dinov2-small", "dinov2-large"]
    scale = meta["by_score"]["colorscale"]
    # brown up to the midpoint, teal above it: the midpoint sits at the best minus the band
    f = (0.76 - noise.band - 0.70) / (0.76 - 0.70)
    assert scale[2][0] == pytest.approx(f) and scale[2][1] == class_analysis.BROWNS[-1]
    assert scale[3][1] == class_analysis.TEALS[0]
    # a filter on one encoder keeps the trials within the noise of that encoder's best
    kept = meta["kept_by_code"]
    assert kept[0][0] == pytest.approx(max(0.74 - noise.band, 0.70))
    assert kept[1][0] == pytest.approx(0.76 - noise.band)
    assert meta["score_axis"] == 2


def test_a_run_has_loss_curves_only_when_it_logs_both_as_numbers():
    both = {"train_loss": [1.0, 0.5], "test_loss": [1.1, 0.9]}
    assert class_analysis.loss_curves(both) == ([1.0, 0.5], [1.1, 0.9])
    assert class_analysis.loss_curves({"train_loss": [1.0]}) is None
    assert class_analysis.loss_curves({"train_loss": [1.0], "test_loss": ["x"]}) is None
    assert class_analysis.loss_curves({"train_loss": [], "test_loss": []}) is None


def test_the_median_curve_stops_where_too_few_curves_reach():
    n = class_analysis.MIN_CURVES
    curves = [[1.0, 0.5, 0.2]] * n + [[3.0]]
    # the first epoch has every curve, the next two have the n long ones
    assert class_analysis.median_curve(curves) == [1.0, 0.5, 0.2]
    assert class_analysis.median_curve(curves[: n - 1]) == []


def idea_row(idea, axes, author="alice"):
    return class_analysis.SimpleNamespace(
        idea=idea, axes=axes, author=author, name="an-idea", student=author
    )


@pytest.mark.parametrize(
    ("idea", "axes", "tags"),
    [
        ("Helpt een dieper hoofd?", ["n_hidden_layers"], ["depth"]),
        ("Does a deeper and wider head help?", ["n_hidden_layers", "hidden_dim"], ["depth", "width"]),
        # a word the study did not search is no tag: the axes it searched are
        ("Does a deeper head help?", ["dropout"], ["dropout"]),
        ("Fusing two encoders", ["encoder", "head"], ["two encoders"]),
        ("What does the head need?", ["n_hidden_layers", "hidden_dim", "dropout", "norm", "skip"], ["whole head"]),
    ],
)  # fmt: skip
def test_an_idea_is_tagged_by_the_words_of_its_question_that_its_study_searched(
    idea, axes, tags
):
    df = class_analysis.pd.DataFrame()
    assert class_analysis.idea_tags(df, idea_row(idea, axes)) == tags


def test_the_handed_out_idea_is_tagged_as_the_handed_out_search():
    df = class_analysis.pd.DataFrame()
    row = idea_row("baseline", [], author=class_analysis.COURSE)
    assert class_analysis.idea_tags(df, row) == ["handed-out search"]


def test_the_cost_figure_carries_each_trial_with_its_tags_and_head():
    noise = class_analysis.Noise(center=0.70, spread=0.005, n=10, dataset="dtd")
    trials = class_analysis.pd.DataFrame({
        "score": [0.72, 0.75, 0.69], "flops": [1e10, 3e10, 2e10],
        "p_encoder": ["dinov2-small", "dinov2-large", "dinov2-small"],
        "p_n_hidden_layers": [1, 0, 1], "student": ["alice", "bob", "carol"],
        "idea_name": ["deep-idea", "", "deep-idea"],
    })  # fmt: skip
    fig, below = class_analysis.fig_cost(
        trials, noise, "accuracy", {"deep-idea": ["depth", "dropout"]}
    )
    assert below == 1
    points = fig.layout.meta["points"]
    assert points[0] == [
        1e10,
        0.72,
        "dinov2-small",
        "alice",
        "deep-idea",
        "depth, dropout",
        "hidden layers",
    ]
    assert points[1][5:] == ["handed-out search", "linear"]
    assert fig.layout.meta["near"] == class_analysis.NEAR
