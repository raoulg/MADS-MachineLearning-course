deep learning course
==============================

This course is part of a series of modules for data science.
This course assumes you have done the introduction in Python and something similar to the Data Analyses & Visualisation course https://github.com/raoulg/MADS-DAV

For this project you will need some dependencies.
The project uses python 3.10, and dependencies are defined within the `pyproject.toml` file. You will also find `reuirements.lock` files, but they are generated for a Mac so they will miss cuda specific dependencies.

The lessons can be found inside the `notebooks`folder.
The source code for the lessons can be found in the `src`folder.


Project Organization
------------

    ├── README.md          <- This file
    ├── .gitignore         <- Stuff not to add to git
    ├── .lefthook.yml      <- Config file for lefthook
    ├── pyproject.toml     <- Human readable file. This specifies the libraries I installed to
    |                         let the code run, and their versions.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The processed datasets
    │   └── raw            <- The original, raw data
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is xx_name_of_module.ipynb where
    │                         xx is the number of the lesson
    ├── presentations      <- Contains all powerpoint presentations in .pdf format
    ├── references         <- Some background information on codestyle, and a courseguide
    │
    ├── reports            <- Generated analysis like PDF, LaTeX, etc.
       └── figures         <- Generated graphics and figures to be used in reporting

--------

The `.lefthook.yml` file is used by [lefthook](https://github.com/evilmartians/lefthook), and lints & cleans the code before I commit it. Because as a student you probably dont commit things, you can ignore it.

I have separated the management of datasets and the trainingloop code. You will find them as dependencies in the project:
- https://github.com/raoulg/mads_datasets
- https://github.com/raoulg/mltrainer

Both of these will be used a lot in the notebooks; by separating them it is easier for students to use the code in your own repositories.

# Installation

## install python with rye
1. watch the [introduction video about rye](https://rye.astral.sh/guide/)
2. You skipped the video, right? Now go back to 1. and actually watch it. I'll wait.
3. install [rye](https://rye.astral.sh/) with `curl -sSf https://rye.astral.sh/get | bash`

run through the installer like this:
- platform linux: yes
- preferred package installer: uv
- Run a Python installed and managed by Rye
- which version of python should be used as default: 3.10
- should the installer add Rye to PATH via .profile? : y
- run in the cli: `source "$HOME/.rye/env"`

## add the git repo
run in the cli:

`git clone https://github.com/raoulg/MADS-MachineLearning-course.git`

## add your username and email to git
1. `git config --global user.name "Mona Lisa"`
2. `git config --global user.email "m.lisa@pisa.com"`

## install all dependencies
1. `cd MADS-MachineLearning-course/`
2. `rye sync`

## add your own ssh key
1. copy your local ssh key, see [github docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
2. `cd ~/.ssh`
3. `nano authorized_keys`
copy paste your key on the second line (leave the first key there) and save the file, then exit
4. check with `cat authorized_keys` that your key is added.

