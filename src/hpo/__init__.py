"""The course's own package for the exercise of every lesson: a study over a design of the
student's own, logged so the class can build on it.

It is four layers. Each imports from the layers below it only, and says in its
`__init__` what it offers:

    exercise   Exercise, Harness       the study flow, the same in every lesson
    record     StudyLog, Idea, Store   how a run reaches the class store
    search     Space, Design, Budget   what a study searches, and how much of it
    measure    Training, Confusion     what one run measures

A study flows through them:

    Questions + Space + Design -> Exercise -> Harness.data -> Training.fit -> Fit
    Fit (Confusion) -> Repeats -> RunRecord -> StudyLog -> the store
"""
