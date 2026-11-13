deep learning course
==============================

This course is part of a series of modules for data science.
This course assumes you have a decent basis in Python and experience with something similar to this Data Analyses & Visualisation course https://github.com/raoulg/MADS-DAV


The lessons can be found inside the `notebooks` folder.

The book we will be using is Understanding Deep Learning. It is available as pdf here: https://udlbook.github.io/udlbook/ but it is highly recommended to buy the book.

To help you track your study progress, I have created this https://learn.pttrn.io web-app. Sign up for the `Machine Learning` course. There you will find an overview of everything you can watch, listen and read for every lesson and check the material when you are done.


Project Organization
------------

    ├── README.md          <- This file
    ├── .gitignore         <- Stuff not to add to git
    ├── .lefthook.yml      <- Config file for lefthook
    ├── pyproject.toml     <- Human readable file. This specifies the libraries I installed to
    |                         let the code run, and their versions.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The processed datasets
    │   └── raw            <- The original, raw data
    │
    ├── docs               <- How-tos: submitting your runs, the run protocol, publishing caches
    ├── notebooks          <- Jupyter notebooks, one folder per lesson
    ├── planning           <- The course's plan: principles, curriculum, evidence
    ├── presentations      <- The presentations, in .html and .pdf format
    ├── references         <- background information, and the setup guides
    |  └── leerdoelen      <- Learning goals per lesson, including pages to read and videos to watch
    │
    ├── reports
    │   └── figures        <- Figures
    └── scripts            <- Building and publishing caches, submitting and collecting class runs

--------

The project uses Python `>=3.12,<3.15` (`requires-python` in `pyproject.toml`), and all dependencies are defined within the `pyproject.toml` file.

The `.lefthook.yml` file is used by [lefthook](https://github.com/evilmartians/lefthook): before every commit it formats, lints and type-checks the code, clears notebook outputs and runs the staged notebooks. The setup below installs it for you, and explains each hook.

Data, representations and the training loop are separated out. You will find them as dependencies in the project:
- [huggingface `datasets`](https://huggingface.co/docs/datasets) — every dataset in the course is loaded from the hub with `load_dataset`. Nothing is downloaded by hand.
- https://github.com/raoulg/vectormesh — vectorizers, the `VectorCache` artefact format, the small shape-checked components you compose into a pipeline, and the `Trainer` that trains them

These will be used a lot in the notebooks; by separating them it is easier to use the code in your own repositories.

In addition to that, you can consider `vectormesh` as "extra material": if you are already more experienced in programming, `BaseVectorizer` and `VectorCache` are worth reading. They are short, and they are a good example of two ideas the course cares about — writing a tensor's shape into its type signature, and treating a computed artefact as a deliverable with a documented contract (`metadata.json`) rather than as a temp file.

In addition to that, there is a [codestyle repo](https://github.com/raoulg/codestyle) that covers most of the codestyle guidelines for the course. You will be expected to follow the codestyle guidelines for your own coding.

# Setup

The setup is meant to be done together with an assistant (Claude Code, Cursor, Codex). Hand it one of these URLs and ask it to walk you through; it takes one step at a time, explains why each tool is there, and keeps track of where you are between sessions.

Start with the local setup. If you did the setup of the DAV course, the guide first asks which topics you already have, so you only do what is left.

## local
The tools on the machine you write code on (git and GitHub, uv, an editor, the course repository, the pre-commit hooks, Atlas, the coaches, and the class runs repository):

```
https://raw.githubusercontent.com/raoulg/MADS-MachineLearning-course/master/references/tooling-setup.md
```

## VM (optional)
Most of you are better off on your own machine, which is usually faster than a VM. A VM is worth requesting when your laptop cannot run the course, for instance because it is managed by your organisation and does not let you install the tools. The lab guide covers requesting a VM and setting it up. Only if you already received a VM invitation by e-mail, do this guide first:

```
https://raw.githubusercontent.com/raoulg/MADS-MachineLearning-course/master/references/lab-setup.md
```

Two rules that matter from day one:

- **Work on your own branch, never on `master`**, and copy a notebook under a new name before editing it. `master` receives new material every week; a renamed file never conflicts with it.
- **Python and packages come from uv only.** No pip, no conda, no separately installed Python.

New to the tools? [uv projects](https://docs.astral.sh/uv/guides/projects/), a [git tutorial](https://www.w3schools.com/git/git_getstarted.asp), and the [command line for beginners](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview).

## Where the course runs
The course is built to run on a laptop CPU: Linux, Windows (via git bash / WSL) and Apple Silicon (M-series) Macs with macOS 14 or newer (the torch wheel the course uses needs it; on an older macOS, update it or use a VM), on Python `>=3.12,<3.15`. A VM is optional: useful for a long training run that should not tie up your laptop, or when your laptop is old or short on memory.

**Intel Macs (x86_64) are not supported**: `torch` and `ray` publish no macOS x86_64 wheels for the versions the course uses. On an Intel Mac, use a VM and follow the VM guide above (`references/lab-setup.md`).