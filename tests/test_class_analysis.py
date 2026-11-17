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
    assert "plotly-graph-div" in class_analysis.build_page(df, 1)


def test_a_label_plotly_draws_holds_no_tag_of_a_student():
    name = "<a href='https://example.org'>click</a>"
    df = class_analysis.runs_frame(
        [sub("mallory", "lesson1-notebook-runs", name, "baseline", 0.8, n_params=10)]
    )
    [trace] = [t for t in class_analysis.fig_architectures(df).data if len(t.x)]
    assert "<" not in trace.text[0] and "&lt;a href" in trace.text[0]


@pytest.mark.parametrize(
    ("sigma", "word"),
    [(2.0, "better"), (1.9, "inside"), (-1.9, "inside"), (-2.0, "worse"),
     (None, "unmeasured"), (math.nan, "unmeasured")],
)  # fmt: skip
def test_a_verdict_is_two_sigmas_either_way(sigma, word):
    assert class_analysis.verdict(sigma) == word


def test_a_parent_that_is_not_submitted_sits_left_of_its_child_inside_the_picture():
    df = class_analysis.runs_frame(
        [with_idea(sub("bob", "lesson1-study-bob-x", "trial-00", "trial", 0.8),
                   idea_attrs("bob", "an idea on a parent nobody sent", ["plaid-eagle-75"]))]
    )  # fmt: skip
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    [name] = ids.name
    place = class_analysis.idea_layout(ids)
    assert place["plaid-eagle-75"][0] == 0 and place[name][0] == 1


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
    assert "handed-out baseline, macro_recall" in page
    assert "Every figure reads macro_recall" in page


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
    page = class_analysis.ideas_note(ids, {"blissful-bee-18": 1})
    assert "an idea of lesson 1" in page


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


def test_an_arrow_carries_the_childs_winner_on_its_parents_and_says_whether_it_counts():
    runs, named = rehearsal()
    df = class_analysis.runs_frame(runs)
    eds = class_analysis.edges(class_analysis.ideas(df, class_analysis.comparisons(df)))
    arrow = eds.set_index(["parent", "child"])
    se = math.sqrt(2 * 0.01**2 / 3)  # two winners of three repeats, spread 0.01

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


GHOST_CHILD = idea_attrs("dave", "on a parent nobody sent", ["plaid-eagle-75"])


def test_an_arrow_between_datasets_or_metrics_or_from_a_lost_parent_has_no_number():
    alice = idea_attrs("alice", "does depth still help", [ROOT["idea_name"]])
    on_beats = idea_attrs("bob", "depth on another dataset", [alice["idea_name"]])
    on_recall = idea_attrs("carol", "depth read on recall", [alice["idea_name"]])
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
    eds = class_analysis.edges(class_analysis.ideas(df, class_analysis.comparisons(df)))
    why = dict(zip(eds.child, eds.why, strict=True))
    assert not eds.number.any()
    assert why[on_beats["idea_name"]].startswith("different datasets")
    assert why[on_recall["idea_name"]].startswith("different metrics")
    assert "not among the submissions" in why[GHOST_CHILD["idea_name"]]


def test_the_window_shows_the_idea_its_parents_and_the_class_faded():
    runs, named = rehearsal()
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    eds = class_analysis.edges(ids)
    bars = class_analysis.gain_rows(ids, eds, named["c2"]["idea_name"])
    assert bars[0]["label"] == "vs its baseline"
    assert sorted(b["label"] for b in bars[1:3]) == ["vs b", "vs c"]
    assert bars[0]["value"] == pytest.approx(10.0) and not bars[0]["faded"]
    assert [b["faded"] for b in bars[3:]] == [True, True, True]  # the other arrows
    everyone = class_analysis.gain_rows(ids, eds)
    assert len(everyone) == 5 and not any(b["faded"] for b in everyone)
    ps = class_analysis.panels(df, ids, eds)
    assert set(ps) == {"", *ids.name}
    assert "expected: c-both pays" in ps[named["c2"]["idea_name"]]["head"]


def test_the_window_draws_the_trials_of_the_study_with_the_best_finished_one_as_winner():
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
    values, labels = class_analysis.scaled(trials, ["hidden_dim", "lr"])
    assert len(values) == 2  # the finished trials
    assert [a for a, _, _ in labels] == ["hidden_dim", "lr", "score"]
    assert values.lr.tolist() == [1.0, 0.0] and values.hidden_dim.tolist() == [0.0, 1.0]
    one = class_analysis.scaled(trials[trials.winner], ["lr"])[0]
    assert one.lr.tolist() == [0.5]  # one trial sits in the middle
    assert class_analysis.shared_axes(ids) == ["hidden_dim", "lr"]


def test_the_tiles_count_ideas_rounds_and_building_on_a_classmate():
    runs, _ = rehearsal()
    df = class_analysis.runs_frame(runs)
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    tiles = class_analysis.class_tiles(df, ids, class_analysis.edges(ids), "accuracy")
    assert "<b>6</b><span>ideas, in 3 rounds</span>" in tiles
    # b1 on a1, c1 on b2, c2 on b1, d1 on c1; c2 on c's own c1 is no classmate's
    assert "<b>4</b><span>times built on a classmate</span>" in tiles
    assert "<b>0.800</b><span>best so far (baseline 0.700)</span>" in tiles


def test_a_class_with_only_studies_of_its_own_says_what_is_missing():
    alice = idea_attrs("alice", "does depth still help", [ROOT["idea_name"]])
    df = class_analysis.runs_frame(study_runs("alice", "alice-a", alice, 0.7, 0.8))
    page = class_analysis.build_page(df, 1)
    assert page.count("No run of the handed-out notebook was submitted") == 3
    assert "id='panels'" in page and "not among the submissions" in page


def test_the_demo_keeps_a_merge_a_missing_parent_and_curves():
    from class_demo import demo_submissions  # ty: ignore[import-not-found]

    df = class_analysis.runs_frame(demo_submissions())
    ids = class_analysis.ideas(df, class_analysis.comparisons(df))
    assert (ids.parents.map(len) == 2).any()
    assert ids.missing.map(bool).any()
    assert (ids[ids.author != "course"].expects != "").all()
    assert df[df.role == "trial"].curve.map(bool).all()
    assert class_analysis.edges(ids).number.any()
