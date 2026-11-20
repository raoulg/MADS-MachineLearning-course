---
name: lesson-design
description: How course material of MADS-ML is designed, built and checked - a notebook or a section of one. Use whenever a notebook of the course is written, rewritten or reviewed ("herschrijf notebook 2.2", "review dit notebook", "is dit begrijpelijk voor een student", "maak een notebook over X"), and before adding a section, a model or an experiment to existing material. Learning goals first, then the material, then a fresh agent that reads it as a student who knows only the earlier lessons; what it cannot learn goes back to the designer. Not for a lesson's slides (that is the slides skill) or a student's own study (that is the your-study skill).
---

# Lesson design

Material is written for a student who knows the earlier lessons and nothing else. The
designer knows too much to see what is missing, so somebody who did not write it reads
it as that student, and the material is done when that reader can learn the goals from
it. This is the board-and-workers loop with learning goals in place of requirements and
a student in place of a code reviewer.

| Role | Does | Never |
|---|---|---|
| **Main session** | The goal map, every decision on the material, the triage of findings, running the notebook | Accepts material on the designer's word |
| **Designer** (Agent) | Writes or rewrites one notebook against its goal map and `references/design-rules.md` | Adds goals, drops goals, edits files outside its brief |
| **Student reader** (fresh Agent each round) | Reads the executed notebook as the student of `references/student-reader.md` | Sees the designer's reasoning or an earlier review |
| **Method reviewer** (fresh Agent) | Checks every experiment and claim with `references/method-reviewer.md` | Judges the prose |
| **Raoul** | Decides what is taught, which goals a lesson has, and what is cut | |

## The loop

1. **Entry profile.** `docs/lesson-design/lesson-<N-1>-exit.md`: what a student entering
   lesson N has actually been taught, per concept and per term, with where. A goal in
   `presentations/leerdoelen/les<N-1>.md` that no notebook or slide teaches is not
   known. Build it once per lesson from the notebooks and slides; refresh it when they
   change.
2. **Goal map.** `docs/lesson-design/lesson-<N>-goals.md`, from `references/goal-map.md`:
   per notebook, the goals it teaches, and per goal what the student can say or do
   afterwards, the new terms it introduces in the order they appear, the picture that
   carries it (a slide or a figure in the notebook), and what from the entry profile it
   leans on. The slides of the lesson are part of the map: say per goal whether the
   slide, the notebook or both carry it. Show Raoul the map before material is written
   against it; the goals are his.
3. **Design.** A designer writes the notebook against the map and the design rules.
   The notebook runs end to end (with `param` for duration) before it goes to a reader.
4. **Run.** The main session executes the notebook with the full parameters and keeps
   the executed copy; a reader reads outputs, not code that might produce them.
5. **Read as a student.** A fresh student reader gets the entry profile, the goal map's
   list of goals for this notebook (the goals only, not the designer's plan), the slide
   text of the lesson, and the executed notebook. In parallel a fresh method reviewer
   gets the executed notebook and the goal map.
6. **Triage.** The main session writes the round to
   `docs/lesson-design/reviews/lesson-<N>-round-<r>.md` (per place in the notebook: what
   the reader and the reviewer found) and sorts the findings: a blocking finding (the student
   cannot learn a goal, or a claim does not follow from the output) goes back to the
   designer with SendMessage; a finding about a goal itself goes to Raoul; the rest is
   decided here.
7. **Again**, with a new reader, until a round has no blocking finding. After three
   rounds with blocking findings, stop and show Raoul what keeps failing: the goal or
   the notebook's plan is wrong, not the wording.
8. **Close.** `uv run notebooktester <notebook> -t 300 -f` passes; the goal map records
   which round passed and what was accepted as it stands.

## What the main session checks itself

- Every goal in the map has a cell where it is taught and a cell where the student does
  something with it (predicts, reads a result, changes a value).
- Every model in the notebook has a reason the student can state before it is trained.
- A reader's "I understand" is a claim too: its answers to the goal questions are the
  evidence. Read them.
