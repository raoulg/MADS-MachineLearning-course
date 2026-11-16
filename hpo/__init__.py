"""hpo, hyperparameter optimisation: the course's own package, shared by every lesson.

- `hpo.optunalog`: the runs of an exercise in Optuna, with their idea, objective and
  budget, in the shape the class run protocol asks.
- `hpo.metrics`: every metric a classification run logs, from one confusion matrix.
- `hpo.designs`: optional stages (`@option`) and the check that a space searches them.
- `hpo.exercise`: the study of an exercise, given a lesson's `Harness`.
"""
