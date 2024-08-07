deep learning course
==============================

This course is part of a series of modules for data science.
This course assumes you have done the introduction in Python and something similar to the Data Analyses & Visualisation course https://github.com/raoulg/MADS-DAV

For this project you will need some dependencies.
The project uses python 3.10, and all dependencies can be installed with `pdm` if you are using that with `pdm install`.

The lessons can be found inside the `notebooks`folder.
The source code for the lessons can be found in the `src`folder.


Project Organization
------------

    ├── README.md          <- This file
    ├── .gitignore         <- Stuff not to add to git
    ├── .lefthook.yml      <- Config file for lefthook
    ├── pyproject.toml     <- Human readable file. This specifies the libraries I installed to
    |                         let the code run, and their versions.
    ├── pmd.lock           <- Computer readable file that manages the dependencies of the libraries
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The processed datasets
    │   └── raw            <- The original, raw data
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is xx_name_of_module.ipynb where
    │                         xx is the number of the lesson
    │
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

