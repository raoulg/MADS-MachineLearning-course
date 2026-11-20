# The goal map

`docs/lesson-design/lesson-<N>-goals.md`. The goals themselves stand in
`presentations/leerdoelen/les<N>.md`; the map says how the material teaches them.

```markdown
# Lesson <N>: goal map

Entry profile: `docs/lesson-design/lesson-<N-1>-exit.md`.

## Notebook <N>.<k> <title>

The student ends this notebook able to: <one or two sentences, the student's words>.

| goal | afterwards the student can | carried by | leans on |
|---|---|---|---|
| 2.5 | say what a GRU carries from step to step, and why its last state can feed a head | slide 10, figure in 2.2.2 | 1.1 shapes, 1.12 reading a chain of blocks |

### New terms, in the order they appear

| term | first cell | explained as |
|---|---|---|
| state | 2.2.2 | the vector a recurrent layer keeps and updates at every step of the sequence |

### Models

| model | why the student meets it here | diagram |
|---|---|---|

### Experiments

| question | arms | what decides it | training length and why |
|---|---|---|---|

### Review rounds

| round | reader | method | blocking left | accepted as it stands |
|---|---|---|---|---|
```
