There are many ways to set up a python environment. The problem is, this can
become a mess very quickly...

# 1 Python
We will setup the management of python with `pyenv`. Installing pyenv works
slightly differently on different machines. Because listing all options is
already done on the page of the pyenv development site, we will simply point you
to those instructions.

## 1.1 Dependencies
First, make sure you have installed the [dependencies](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)

## 1.2 UNIX
You can find instructions for different UNIX (macOS or Linux) operating systems [here](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)

After that, follow [this](https://github.com/pyenv/pyenv#homebrew-in-macos) for
mac (where we recommend brew)
or [this](https://github.com/pyenv/pyenv#basic-github-checkout) for linux
They are almost the same, except for the first step. Configure your shell,
depending on the shell you use.

## 1.3 Windows
Don't use windows for coding.
If you must, you can install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) which works very nice, and you can connect [VScode](https://code.visualstudio.com/blogs/2019/09/03/wsl2) to it.

## 1.4 Install and switch version

Now, you can check the versions you have installed with `pyenv versions`. You
probably only have the system version of python. This could be python 2, or
python 3. To check your version, excecute `python --version` in your shell.

Installing a new version is as simple as:

`pynenv install 3.9.7`

This will install version 3.9.7, no suprise there...
Checking `pyenv versions` will show the newly added python version.

This is really the cleanest way to manage different versions of python. If you
want to list some available versions, you can use this:
`pyenv install --list | grep " 3\.[891]"`
It will show the list of available versions to install, and filter version 3.8,
3.8 or 3.10 and above.

It searches for a `.python-version` file in your current folder, or in its
parent folders. This file contains a string defining the version to use. The
command `pyenv local 3.9.7` will create this file for you.

# 2 Virtual environments
There are many ways to manage packages. Most common are `pip` and `conda`.
However, they can cause annoying problems. There might be situations where pip or conda is preferred, but as a default, we recommend using `pdm`.

## 2.1 PDM
The basic installation guide can be found
[here](https://pdm-project.org/en/latest/)


## 2.2 Setting up an environment

Now, you can go to a folder where you want to start your new project.
Inside that folder, you can switch to the correct python version you would like
to use with pyenv (eg `pyenv local 3.9.7`) and after that you type `pdm init`
and poetry will guide you through an interactive menu to set up your
`pyproject.toml` file

## 2.3 pyproject.toml file

The parentfolder of this lesson contains a pyproject.toml file. Have a look at
it. You will find a section with dependencies used in the project (like numpy,
seaborn and scikit-learn).

Adding new dependencies can be done with `pdm add`, eg `pdm add scipy`.

## 2.4 dev dependencies
What is very usefull, is that we can specify dependencies we will only use for
development (like libraries for linting and hypertuning for a model). For
example, if we want to install `black` for linting as a dev-package, we just add
the `-dG` postfix (dG for dev group), eg `pmd add -dG black`.

## 2.5 install
Installing all the libraries needed can be done by navigating into a folder with
a pyproject.toml file, and just run `pdm install`. pdm will install
everything asynchronously, and not sequentially like pip or conda. This makes it
a lot faster.

# 3 Activate
Activating an environment can be done in the shell by typing `eval $(pmd venv activate)`, and
the environment will be activated.
