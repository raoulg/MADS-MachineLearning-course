# Design rules for a notebook

The designer's brief carries this file. Each rule answers a question the student
otherwise asks: what is this, why now, what do I look at, what did it show.

## The opening

- The first cell a student reads says what they will be able to do at the end, in their
  words, and why it matters for the class problem. A table of goal numbers is not that.
- The first code cell does one thing the student can name: load the data and show it.
  It does not define a model, and it does not train one.

## A new idea

- Introduce before use. Every term the entry profile does not list is explained the
  first time it appears, in plain words, before code uses it. Not "the state", but what
  is carried, from where to where, and why a kernel cannot.
- Problem first. Show the limit of what the student already has (the kernel cannot
  see beyond its field; a deep plain stack stops learning), then the block that answers
  it. A block that arrives without a problem is a name to memorise.
- One new idea per section. A section that introduces a state and a gate and a highway
  introduces none of them.
- Name where an idea comes from when it helps the student place it (ResNet, 2015, for
  image models), not as a citation exercise.

## A model

- Before the code: what it is for, a block diagram with the tensor shape on every arrow
  (an inline SVG in the markdown cell, or a figure from `diagrams.py`), and what the
  student should expect. Draw it the way lesson 1 drew architectures.
- Shapes are explained, not only printed. `(B, 46, 32) -> (B, 46, 64) -> (B, 64)` gets
  a sentence per arrow: what each axis is, what changed, and why (`[:, -1]` keeps the
  state after the last step, because that one has seen the whole beat).
- A model is defined once and reused by name. A model from an earlier notebook is
  imported or shown with a pointer to where it was explained, never redefined in
  silence.

## An experiment

- Question, expectation, setup, result, reading: in that order, each in its own place.
  The student writes the expectation down before the cell that answers it runs.
- The setup is fair, and the notebook says why. Models of different depth or size get a
  training length that lets each of them converge, and the notebook shows the loss
  curves, so the student sees that it did. A comparison where one arm is still learning
  is not a comparison.
- A baseline the student already has is reused and named ("the kernel stack of 2.1.5,
  0.71 ± 0.02"), not recomputed without a word. If it is recomputed, the notebook says
  why (another split, another budget).
- The reading points at the number or the curve that carries the conclusion, and says
  what would have meant the opposite. A conclusion the output does not show is not
  written.
- A control differs from its arm in one factor, or the reading names every factor it
  also changes (a wider kernel also has more weights).
- A baseline for a picture of what a trained block does keeps everything else trained: a
  random gate on the trained features, not the untrained model.
- A figure cited for a mechanism must be able to tell that mechanism apart from the
  obvious other one; otherwise add the measure that does, or say what it cannot show.
- Every expensive constant uses `param(full, test=...)`.

## Pictures

- Every goal has a picture that carries it, in the slides or in the notebook, and the
  notebook refers to the slide by number when the slide carries it.
- A figure has a title that says what to look at, axes with units, and a sentence under
  it that says what it shows.

## The close

- "What you now know" restates each goal as something the student can now say, and
  matches what the notebook taught. A claim there that no cell supports is cut.
