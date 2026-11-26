# Your study: instructions for a coding agent

A student pointed you here to walk the loop of a study with them: one idea, a design that
can test it, a search, and a result the class can build on. The loop has nine steps and
is the same in every lesson; it lives in the lesson's `exercises/` folder. The student
reads the same steps, with what each is for, on
[Your study, as a loop](https://raoulg.github.io/MADS-MachineLearning-course/your-study.html)
(`presentations/your-study.html`); what a lesson brings to it (its data, its baseline, its
exercise notebook, its harness) stands there under "Per lesson", `#lesson-1`,
`#lesson-2` and `#lesson-3`. Run every command from the root of the course repository (the directory with
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
| run the study | 3 | `make study LESSON=3` | `uv run notebooks/3_scaling_and_search/exercises/study.py` |
| read the student's own runs | any | `make optuna-dashboard` | `uv run optuna-dashboard sqlite:///artefacts/optuna/studies.db --artifact-dir artefacts/optuna/artifacts --port 8088` |
| build and open the student's page | any | `make page LESSON=<N>` | `uv run scripts/study_page.py --lesson <N> --open` |
| hand in | 1 | `make share LESSON=1` | `uv run scripts/submit_runs.py --lesson 1` |
| hand in | 2 | `make share LESSON=2` | `uv run scripts/submit_runs.py --lesson 2` |
| hand in | 3 | `make share LESSON=3` | `uv run scripts/submit_runs.py --lesson 3` |
| the class's ideas and runs | 1 | `make dashboard LESSON=1` | `uv run scripts/class_analysis.py --lesson 1 --open --no-pull`, then `uv run scripts/collect_runs.py --lesson 1 --no-pull --ui` |
| the class's ideas and runs | 2 | `make dashboard LESSON=2` | `uv run scripts/class_analysis.py --lesson 2 --open --no-pull`, then `uv run scripts/collect_runs.py --lesson 2 --no-pull --ui` |
| the class's ideas and runs | 3 | `make dashboard LESSON=3` | `uv run scripts/class_analysis.py --lesson 3 --open --no-pull`, then `uv run scripts/collect_runs.py --lesson 3 --no-pull --ui` |

`make optuna-dashboard` serves on `http://127.0.0.1:8088` (`OPTUNA_PORT=<port>` for
another port); its store exists once Part 1 of the lesson's exercise notebook has run.
`make share` takes `ARGS=--list` to show what would be handed in, and `make dashboard`
takes `METRIC=<name>` to read the class page on another logged metric (plain form:
`--metric <name>` on `class_analysis.py`). `make dashboard` reads the class's runs as
they were last fetched; `PULL=1` fetches the newest first (the first run fetches on its
own).

The lesson folders are `notebooks/1_representation_and_head/exercises/`,
`notebooks/2_kernels_state_and_gates/exercises/` and
`notebooks/3_scaling_and_search/exercises/`. The exercise notebook is
`05_exercises.ipynb` in lesson 1, `04_exercise.ipynb` in lesson 2 and
`05_exercise.ipynb` in lesson 3.

Lesson 3 adds two things to the loop. Every run logs what one image costs it,
`flops_per_image` and `latency_us`, beside `n_params`, so step 6 reads accuracy against
FLOPs as a Pareto front, not only the best trial (`rows_of` and `plot_front` in
`harness.py` draw it from the store). And a study of an architecture of the student's own
has a control of equal compute beside it: the plain head at the same FLOPs per image,
among the trials of the same space (`control_width` in `designs.py` gives its width from
the design's FLOPs; Part 2 of `05_exercise.ipynb` gives the recipe).

## Before the loop: Part 1 of the exercise notebook

Part 1 of the lesson's exercise notebook walks steps 3 to 6 once, with a design and a
space that are given, and its runs are handed in with the study's in step 8. It measures
the baseline and what one trial costs on the student's machine, which steps 3 and 4 use.
Ask whether it has run on this machine; if not, the student runs it first.

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
atlas-code-nightly view notebooks/3_scaling_and_search/exercises        # lesson 3
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
uv run notebooks/3_scaling_and_search/exercises/designs.py         # lesson 3
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

**Ask:** what did each axis do to the objective and to `train_seconds`? How far is the
winner from the baseline, in points, beside the spread between the repeats of each? What
does that say about what they wrote in `EXPECTS`?

```bash
make optuna-dashboard
```

Every run logs more than the objective, for a question asked later.

## 7. Explain

The student builds their page and you interview them about it, so that what stands on
the page is what they can say themselves. The page is what the teacher reads in class,
walking past, in three minutes.

```bash
make page LESSON=<N>
```

It writes `artefacts/study-page/lesson<N>/index.html` and serves it on the student's own
machine until Ctrl-C; the student reads it in the browser. Its tabs sit in a rail on
the left. Overview holds their ideas as a tree of titles, each idea under the one it
builds on, with what its winner moved from the idea it builds on (± its standard error)
and from its own baseline; opened, an idea shows what it expected. No idea is judged:
one that found nothing can teach as much as one that rose. Studies shows
one study at a time, picked by its title: the idea, the architecture, the search space,
the winner next to its baseline, the learning curves. Explore holds every finished trial
of all their studies: the student sets two fields against each other and a third in
colour, picks the studies, and filters (only the heads above 300k parameters, only the
trials above 0.82, only those that logged recall). Scaling sets the score of every trial
against the parameters of the head and against the FLOPs of training it, coloured by
encoder, and shows the training curves (train and test loss) per size of the head. A
view of Explore and every picture of Studies and Scaling can be starred; Starred gathers them all, and a switch on the
Overview shows them there too. Stars are saved in `exercises/stars.json`, with their
filters, so they are there again the next time the page opens; Design shows the design in Atlas. The pen beside the title of an opened idea edits that title
in `page.toml`. `make page LESSON=<N> ARGS=--no-serve` opens the written file instead,
and then a star or a title is not saved. What is the student's own stands in `exercises/page.toml` of the lesson
folder: `message`, `picture`, `surprised` and `next`, and under `[titles]` a title of
their own per study, by the study's name in study.py. The script creates that file with
empty keys once the lesson has runs, and never writes it afterwards; an empty key shows
as an open slot on the page. With no runs of the lesson in the store, `make page` stops
and says to run the study first: that is step 5.

The page is a starting point, not a form. A student who wants to see their runs another way
(another picture, another comparison, a layout that tells their story better) is
encouraged to build their own, better dashboard with Claude: from `scripts/study_page.py`
or from scratch, on the same store (`artefacts/optuna/studies.db`). What keeps it honest
stays: every score next to its baseline, and no axis that mixes two datasets. The student
decides what it shows; you build what they ask for.

This step is about a study of the student's own. When the page shows only the handed-out
study (one idea "handed out" in the tree, no expectation, no design script), the
student has no study of their own yet: say so, and go with them to step 1. Do not
interview them about the handed-out study.

**Ask, one question at a time, and wait for each answer before you ask the next:**

1. Do you understand everything on your own page?
2. Can you say in three sentences what you did, without looking?
3. What did you expect, and what surprised you?
4. What do you want to look into next, and which idea of the class does it build on?
5. Is there one picture that would carry your message, and what would be on it?

When the student cannot explain a part of the page, name that part and send them back to
the step it belongs to: the blocks of the architecture and the Design tab to step 1, the
shapes to step 2, the spread of the baseline to step 3, the axes and ranges and a study's
idea and expectation in the line of ideas to step 4, a refusal or a missing trial to
step 5, the winner against the baseline in points to step 6, and what a study builds on
to step 9. The interview goes on once they can say it. A starred view is the student's own question to
the trials: ask what they see in it, and do not star views for them.

Write `exercises/page.toml` once, after the last question, in the student's own words:

- `message`: what they did, the answer to question 2;
- `surprised`: what they expected against what happened, the answer to question 3;
- `next`: the next idea and the idea of the class it builds on, the answer to question 4;
- `picture`: the file of the picture, below.

Correct spelling and nothing else: do not write the message for them, do not improve or
shorten their sentences, and do not judge the page. Rebuild the page with the same
command and ask the student to read it back. A message of more than three sentences shows
on the page as "N sentences: say it in 3"; ask the student to say it shorter, and write
down what they say.

For the picture, the student says what it shows, and the picture is drawn as a file in
the lesson's `exercises/` folder, never as an answer in the chat. The beelddenker skill
draws it when you have it: it is in your list of skills or it is not. When it is not, say
that the student can install it from https://github.com/raoulg/beelddenker, and meanwhile
draw an `.svg` yourself from what the student said the picture shows. Name the file under
`picture` in `page.toml`, as a path relative to `exercises/`: an `.svg`, a `.png` or an
`.html` fragment. A file outside that folder, or one that does not exist, shows on the
page as a line saying so in the picture's slot. The student decides whether the picture
stays; one they do not want is taken out of `page.toml`.

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
