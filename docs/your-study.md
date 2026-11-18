# Your study: instructions for a coding agent

A student pointed you here to walk the loop of a study with them: one idea, a design that
can test it, a search, and a result the class can build on. The loop has nine steps and
is the same in every lesson; it lives in the lesson's `exercises/` folder. The student
reads the same steps, with what each is for, on
[Your study, as a loop](https://raoulg.github.io/MADS-MachineLearning-course/your-study.html)
(`presentations/your-study.html`); what a lesson brings to it (its data, its baseline, its
exercise notebook, its harness) stands there under "Per lesson", `#lesson-1` and
`#lesson-2`. Run every command from the root of the course repository (the directory with
`pyproject.toml` and `Makefile`).

The student designs and states the hypothesis; you ask first, and write code once they
have said it in their own words.

## The commands

Each action has a `make` form and a plain form for a machine without make (`make --version`
tells you which you have). The plain form is what the `Makefile` recipe runs.

| Action | Lesson | With make | Without make |
| --- | --- | --- | --- |
| run the study | 1 | `make study` or `make study LESSON=1` | `uv run notebooks/1_representation_and_head/exercises/study.py` |
| run the study | 2 | `make study LESSON=2` | `uv run notebooks/2_kernels_state_and_gates/exercises/study.py` |
| read the student's own runs | any | `make optuna-dashboard` | `uv run optuna-dashboard sqlite:///artefacts/optuna/studies.db --artifact-dir artefacts/optuna/artifacts --port 8088` |
| build and open the student's page | any | `make page LESSON=<N>` | `uv run scripts/study_page.py --lesson <N> --open` |
| hand in | 1 | `make share LESSON=1` | `uv run scripts/submit_runs.py --lesson 1` |
| hand in | 2 | `make share LESSON=2` | `uv run scripts/submit_runs.py --lesson 2` |
| the class's ideas and runs | 1 | `make dashboard LESSON=1` | `uv run scripts/class_analysis.py --lesson 1 --open`, then `uv run scripts/collect_runs.py --lesson 1 --no-pull --ui` |
| the class's ideas and runs | 2 | `make dashboard LESSON=2` | `uv run scripts/class_analysis.py --lesson 2 --open`, then `uv run scripts/collect_runs.py --lesson 2 --no-pull --ui` |

`make optuna-dashboard` serves on `http://127.0.0.1:8088` (`OPTUNA_PORT=<port>` for
another port); its store exists once Part 1 of the lesson's exercise notebook has run.
`make share` takes `ARGS=--list` to show what would be handed in, and `make dashboard`
takes `METRIC=<name>` to read the class page on another logged metric (plain form:
`--metric <name>` on `class_analysis.py`).

The lesson folders are `notebooks/1_representation_and_head/exercises/` and
`notebooks/2_kernels_state_and_gates/exercises/`. The exercise notebook is
`05_exercises.ipynb` in lesson 1 and `04_exercise.ipynb` in lesson 2.

## Before the loop: Part 1 of the exercise notebook

Part 1 of the lesson's exercise notebook walks steps 3 to 6 once, with a design and a
space that are given, and its runs are handed in with the study's in step 8. It measures the baseline and what one trial costs on the student's
machine, which steps 3 and 4 use. Ask whether it has run on this machine; if not, the
student runs it first.

## 1. Design

**Ask:** what is the model, as blocks in order? What shape does each block take and give?
Which block may the search leave out? Which idea does this first design build on (the
exercise as handed out, `Idea.handed_out(<N>).name`)?

The design is `designs.py`: a `Config` with the choices, one method per stage, and a
`forward` that calls them in order. Use the vectormesh coach's `/vectormesh:design` to ask
these one stage at a time. The student reads the shapes in atlas before anything trains:

```bash
atlas-code-nightly view notebooks/1_representation_and_head/exercises   # lesson 1
atlas-code-nightly view notebooks/2_kernels_state_and_gates/exercises   # lesson 2
```

## 2. Build

**Ask:** what is each optional stage's axis called? Are the shapes printed the ones the
sketch promised?

An optional stage carries `@option("<axis>")`, has a field in `Config`, and runs under
`if self.config.<axis>:` in `forward`. The file ends in `DESIGN = Design(...)` and
`DESIGN.show(example, ...)`, which builds every combination of the options and prints the
shape after every stage:

```bash
uv run notebooks/1_representation_and_head/exercises/designs.py   # lesson 1
uv run notebooks/2_kernels_state_and_gates/exercises/designs.py    # lesson 2
```

## 3. Baseline

**Ask:** how far apart do two runs of the same linear model land, and what does one trial
cost here? The student reads both from Part 1 of the exercise notebook.

There is no command of its own: `study.py` measures the baseline before every search, in
step 5. A difference inside that spread is not a difference.

## 4. Budget and space

**Ask, before any code:** what is the study called? What does it investigate, in a few
words of their own, and which idea does it build on? What do they expect, and why? Which
metric does the search maximise? Which axes, over which ranges? How many minutes?

The answers become the six settings at the top of `study.py`: `STUDY`, `IDEA`
(`Idea.of(..., builds_on=...)`), `EXPECTS`, `OBJECTIVE`, `SPACE` and `MINUTES`. Every
option of the design needs an axis in `SPACE`, searched or fixed. Write them once the
student has said them; do not pick an idea, a parent or an expectation for them.

## 5. Search

**Ask:** is `study.py` set as they said? If the script refuses, what do they change: the
space, the design, or the name of the study?

```bash
make study LESSON=<N>
```

The script runs an Optuna study over the space and the design, then reruns the winner as
often as the baseline. It refuses a space that leaves out an option or holds an axis that
neither the design nor the trainer knows, and a study that already holds trials of
another space. Put a refusal to the student; do not quietly widen the space.

## 6. Read

**Ask:** what did each axis do to the objective and to `train_seconds`? How many sigmas
is the winner from the baseline? What does that say about what they wrote in `EXPECTS`?

```bash
make optuna-dashboard
```

Inside about two sigmas, the difference is not a result, whatever its sign. Every run logs
more than the objective, for a question asked later.

## 7. Explain

The student builds their page and you interview them about it, so that what stands on
the page is what they can say themselves. The page is what the teacher reads in class,
walking past, in three minutes.

```bash
make page LESSON=<N>
```

It writes `artefacts/study-page/lesson<N>/index.html` and opens it; the student reads it
in the browser. The template fills what the store holds: the line of their ideas, the
architecture, the search space, the winner next to its baseline, and a tab per study.
What is the student's own stands in `exercises/page.toml` of the lesson folder: `message`,
`picture`, `surprised` and `next`. The script creates that file with empty keys once the
lesson has runs, and never writes it afterwards; an empty key shows as an open slot on the
page.

**Ask, one question at a time, and wait for each answer before you ask the next:**

1. Do you understand everything on your own page?
2. Can you say in three sentences what you did, without looking?
3. What did you expect, and what surprised you?
4. What do you want to look into next, and which idea of the class does it build on?
5. Is there one picture that would carry your message, and what would be on it?

Write the student's own words into `exercises/page.toml`: their three sentences into
`message`, what surprised them into `surprised`, what they look into next into `next`.
Rebuild the page with the same command and ask the student to read it back. Correct
spelling and nothing else: do not write the message for them, do not improve their
sentences, and do not judge the page.

When the student cannot explain a part of the page, name that part and send them back to
the step it belongs to: the blocks of the architecture to step 1, the shapes to step 2,
the spread of the baseline to step 3, the axes and ranges to step 4, a refusal or a
missing trial to step 5, the winner in sigmas to step 6. The interview goes on once they
can say it.

For the picture, the student says what it shows, and you draw it with the beelddenker
skill (https://github.com/raoulg/beelddenker) as an `.svg` or an `.html` fragment in the
lesson's `exercises/` folder. Name it under `picture` in `page.toml`, as a path relative
to that folder; a file outside the folder is refused. Rebuild the page. The student
decides whether the picture stays; one they do not want is taken out of `page.toml`.

## 8. Share

**Ask:** is the list of runs that will be handed in the one they mean to hand in?

Handing in follows [`docs/submit-runs.md`](submit-runs.md); go there and follow it. Every
run of the study carries `designs.py` as its design script.

## 9. Build on

**Ask:** which idea of the class does the next one build on, and what do they expect it to
add?

```bash
make dashboard LESSON=<N>
```

The class page shows the class's ideas with their names (`blissful-bee-18`). The name the
student picks goes in `builds_on`, and a new idea is a new `STUDY`: back to step 1. The
student defends the new design without you. The page of a class rehearsal,
`presentations/lesson1-class-rehearsal.html`, shows what a class page looks like after
three rounds.
