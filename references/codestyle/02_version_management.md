# Python version management

![python versions xkcd](https://imgs.xkcd.com/comics/python_environment.png)

Unfortunately, Python is a language that does not have a robust dependency management as a default. The result is that over the years, a lot of different tools have accumulated in different attempts to solve this problem.

While you can get pretty far by at random combining `pip install`, `conda install`, at some point beginners will typically end up with something that is as broken as the above comic.

We are trying to learn you a robust Python style, where you do not learn tools that you will have to unlearn at the moment you are growing into more demanding projects.

To obtain this goal of robustness, we encourage you follow this philosophy:

- manage your Python versions with pyenv
- manage your dependencies with a tool that uses the `pyproject.toml` file, such as pdm or poetry
- for every project, make a virtualenv
- split dependencies into two minimal categories: development and production

### For windows users
Unfortunately, pyenv does not work on windows. However, you can use a port: [pyenv-win](https://github.com/pyenv-win/pyenv-win).


## Python versions with Pyenv

When developing in Python, unfortunately there is not a culture of backwards compatibility. Even though Python 3 is released in 2008, there are still projects that are only compatible with Python 2.7. This is a problem, because Python 2.7 is no longer supported since 2020.

But even within the minor versions, there are problems. If you want to use the latest version of SciPy, you will need at least Python 3.9. In practice, this means you could want to use the latest python 3.11 for your own project, but you might have issues with dependencies and might need to downgrade to 3.10 or even 3.9 to get everything working. Currently, python 3.12 is out but I'm still not able to use it because some essential packages I use are not yet working with 3.12. Unfortunate, we know, but that is why they created new languages like [julia](https://julialang.org/) that have a syntax comparible to Python, but with fixes for some of the most basic problems with python (among which: a 10-100x speedup, native package management that works, no need for a C/C+ backend, etc). [rust](https://www.rust-lang.org/) even guarantees that your code will always compile with new versions of the language.

Anyway, Python is still relevant and we use Python in this course, which means it is a smart idea 💡 to use a python version manager. We recommend [Pyenv](https://github.com/pyenv/pyenv).

To install it, you can use to automatic installer if you are using bash on any unix system:

```bash
curl https://pyenv.run | bash
```

or with homebrew on OS X:

```bash
brew install pyenv
```

and you can always install it manually:

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

After that, you will need to set up your shell to use pyenv. This is different for every shell. In short, this means telling your shell where to find pyenv. Details can be found on the [website](https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv) for every type of shell.

You will also need to add build dependencies if you want to install python versions. This is different for every system, so read [the manual](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) and install what you need.

If you do not understand what happens when you add something to PATH in your shell, you should have read [the previous section on paths](01_path.md). For a bit more thorough review of how pyenv works, read [this tutorial](https://opensource.com/article/20/4/pyenv).
