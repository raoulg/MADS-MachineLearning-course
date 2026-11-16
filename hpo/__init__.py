"""The course's own package for the exercise of every lesson: a study over a design of the
student's own, logged so the class can build on it.

    store         Store: where the studies, the artifacts and the login live
    protocol      the names hpo shares with the class run protocol
    tensors       the shapes of what goes into a model and comes out
    metrics       Confusion -> Scores; Repeats and Scalars over the repeats of a run
    space         Space: the axes a study draws from
    designs       Design: the options of a supergraph
    ideas         Idea: what a study investigates, and what it builds on
    architecture  Architecture: the layer table of a model
    training      Training -> Fit
    runs          RunRecord, StudyLog: a run as the class run protocol writes it
    budget        Budget: minutes into trials
    harness       Harness: what a lesson gives the study
    compare       Comparison: two sets of repeats in sigmas
    exercise      Exercise: the study flow, the same in every lesson
"""

from hpo.architecture import Architecture
from hpo.budget import Budget
from hpo.compare import Comparison
from hpo.designs import Design, option
from hpo.exercise import Exercise, Outcome, Questions
from hpo.harness import Baseline, Data, Harness
from hpo.ideas import Idea, study_name
from hpo.metrics import Confusion, Objective, Repeats, Scalars, Scorer, Scores
from hpo.runs import SESSION, DesignScript, RunRecord, StudyLog
from hpo.space import Choice, Float, Int, Space
from hpo.store import Store
from hpo.training import Curves, Fit, Training, repeat

__all__ = [
    "SESSION",
    "Architecture",
    "Baseline",
    "Budget",
    "Choice",
    "Comparison",
    "Confusion",
    "Curves",
    "Data",
    "Design",
    "DesignScript",
    "Exercise",
    "Fit",
    "Float",
    "Harness",
    "Idea",
    "Int",
    "Objective",
    "Outcome",
    "Questions",
    "Repeats",
    "RunRecord",
    "Scalars",
    "Scorer",
    "Scores",
    "Space",
    "Store",
    "StudyLog",
    "Training",
    "option",
    "repeat",
    "study_name",
]
