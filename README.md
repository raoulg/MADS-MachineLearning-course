deep learning course
==============================

This course is part of a series of modules for data science.
This course assumes you have a decent basis in Python and experience with something similar to this Data Analyses & Visualisation course https://github.com/raoulg/MADS-DAV


The lessons can be found inside the `notebooks`folder.
The source code for the lessons can be found in the `src` folder.

The book we will be using is Understanding Deep Learning. It is available as pdf here: https://udlbook.github.io/udlbook/ but it is highly recommended to buy the book.

To help you track your study progress, I have created this https://learn-79720.web.app/courses web-app. Sign up for the `Machine Learning` course. There you will find an overview of everything you can watch, listen and read for every lesson and check the material when you are done.


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
    ├── references         <- background information
    |  └── codestyle       <- Some code Code style standards
    |  └── leerdoelen      <- Learning goals per lesson, including pages to read and videos to watch
    │
    ├── reports            <- Generated analysis like PDF, LaTeX, etc.
       └── figures         <- Generated graphics and figures to be used in reporting

--------

For this project you will need some dependencies.
The project uses python 3.11 or 3.12, and all dependencies are defined within the `pyproject.toml` file.

The `.lefthook.yml` file is used by [lefthook](https://github.com/evilmartians/lefthook), and automatically lints & cleans the code before I commit it. Because as a student you probably dont commit things, you can ignore it, but you might want to use lefthook in your own repositories.

I have separated the management of datasets and the trainingloop code. You will find them as dependencies in the project:
- https://github.com/raoulg/mads_datasets all code for datasets and dataloaders
- https://github.com/raoulg/mltrainer all code for the training

Both of these dependencies will be used a lot in the notebooks; by separating them it is easier to use the code in your own repositories.
In addition to that, you can consider the packages as "extra material"; the way the pacakges are set up is something you can study if you are already more experienced in programming.

In addition to that, there is a [codestyle repo](https://github.com/raoulg/codestyle) that covers most of the codestyle guidelines for the course. You will be expected to follow the codestyle guidelines for your own coding.

# Installation
The installation guide assumes a UNIX system (os x or linux).
If you have the option to use a VM, see the references folder for lab setups (both for azure and surf, pick the one you are provided with).
For the people that are stuck on a windows machine, please use [git bash](https://gitforwindows.org/) whenever I 
refer to a terminal or cli (command line interface).

## install python with uv
please note that uv might already be installed on your VM.
1. check if uv is already installed with `which uv`. If it returns a location, eg `/Users/user/.cargo/bin/uv`, uv is installed.
2. else, install [uv](https://docs.astral.sh/uv/) with `curl -LsSf https://astral.sh/uv/install.sh | sh` for macOS and Linux. For windows, see the uv docs.
3. start a new terminal and test with `which uv` if the installation was succesful.

## add the git repo
run in the cli:

`git clone https://github.com/raoulg/MADS-MachineLearning-course.git`

## add your username and email to git
1. `git config --global user.name "Mona Lisa"`
2. `git config --global user.email "m.lisa@pisa.com"`

## install all dependencies
1. `cd MADS-MachineLearning-course/`
2. `uv sync`

## add your own ssh key
If you want easy access to the VM, add your ssh key to the VM:
1. copy your local ssh key to the VM, see [github docs](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) and/or the azure manual.
2. `cd ~/.ssh`
3. `nano authorized_keys` (or, [if you know how to use it](https://www.youtube.com/watch?v=m8C0Cq9Uv9o), install and use `nvim`)
copy paste your key to the end of the file (leave existing keys) and save the file, then exit
4. check with `cat authorized_keys` that your key is added.

## Make yourself familiar

Get yourself familiar with the toolbox, if you arent already
- uv: either read the docs [uv projects docs](https://docs.astral.sh/uv/guides/projects/) or watch a video on youtube, eg [uv for python](https://youtu.be/qh98qOND6MI?si=hjFMgpAYaUuV_Hgl)
- git: do a [tutorial](https://www.w3schools.com/git/git_getstarted.asp) or watch a [video](https://youtu.be/r8jQ9hVA2qs?si=2GicBn0xNeG3wACD)
- terminal/unix: get familiar with the terminal and the unix commands, again [video](https://youtu.be/gd7BXuUQ91w?si=9WqWNQk5MAKr-sF1) or some [tutorial](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview)
