## Virtual environments

Now we get to a hot potato of Python: dependency management and virtual environments. Different packages and modules not only depend on different versions of Python [as explained here](02_version_management.md), but often also require (specific versions of) other packages. If these versions do not correspond, there is a good chance that your code will not work.

This is where virtual environments and dependency management comes into play. Virtual environments are project specific. You can think of virtual environments as secluded spaces on your computer, where you can install a specific version of Python and all the packages (running on this specific Python version) you need for a specific project. Like an island with it's own specific (Python) eco-system. This way, you can have multiple projects on your computer, each with their own virtual environment, and each with their own versions of python and corresponding package versions.

For example, you might have an older project that uses `Python 3.7` and the package [`Numpy`](https://numpy.org/devdocs/index.html) installed. The latest version of Numpy that supports `Python 3.7`  is [`Numpy 1.21.6`](https://numpy.org/devdocs/release/1.21.6-notes.html), so you might have that version installed in the virtual environment of this project. But a newer project that you are working on might use `Python 3.10`. `Numpy version 1.21.6` doesn't support `Python 3.10`, so you will need to install [different version of Numpy](https://numpy.org/devdocs/release/1.22.0-notes.html).

If you would install both versions of `Numpy` on your computer, there is a good chance that you will run into problems. A virtual environment solves this problem. You can create a virtual environment for each project, and install the correct versions of Python and `Numpy` in each of them. When you call python and package functions in your code, python will use the versions stored in the virtual environment of the project. This way, you can work on both projects without having to worry about version conflicts.

## Dependency management with Rye

These version requirements of packages in a project, is called dependency management. Python is known for its horrific dependency management. Of the many tools that try to solve this problem, none of them are perfect. The most (in)famous and widely used tool is [`Anaconda`](https://www.anaconda.com/). However, Anaconda is bulky (it adds over 5GB of dependencies to your environment), doesnt follow [PEP standards like the `pyproject.toml` file](https://peps.python.org/pep-0621/) and if you make a venv and export the dependencies, the export is OS specific and wont be reproducable for other OS-es (eg between Linux and Windows) unless you use a workaround to strip the OS-specific parts.

Therefore, we have chosen to use `rye` as our dependency manager. It is not perfect either, but implements some of the best practices available and helps avoid a lot of problems.

#### 0. Python native venvs
With python, you can create virtual environments that isolate your dependencies. It works like this:
To create a virtual environment, open your command prompt or terminal and navigate to the directory where you want to create the virtual environment.
```bash
python -m venv .venv
```
This command will create a new virtual environment in a directory named .venv in your current working directory.

You need to activate the virtual environment to work within it.

```bash
source .venv/bin/activate
```
Once activated, you'll see the virtual environment's name in your command prompt, indicating that you are now working within the virtual environment.
With the virtual environment activated, you can use pythons native `pip` to install packages and dependencies. For example, to install a package named example-package, use:
```bash
pip install example-package
```
To leave the virtual environment and return to your system's Python environment, you can deactivate it. Simply run:
```bash
deactivate
```

If you no longer need the virtual environment, you can delete it. Ensure the virtual environment is deactivated first. Then, you can remove the entire directory:
```bash
rm -rf .venv
```
This example is to show you how this would work with base python. However, we can use PDM to both create a .venv and install the packages from a pyproject.toml file, so you dont need to manage it with pip and write the dependencies down in a requirements.txt file, adding version constraints manually.

#### 1. PDM centralizes dependencies in pyproject.toml

**The first reason is that it uses the `pyproject.toml` file to store dependencies**, according to the [pep 621](https://peps.python.org/pep-0621/) standard for python projects. You can think of a `.toml` file as a bunch of settings and requirements that apply to your project.

Don't worry if you do not understand all the variables defined in the `pyproject.toml` file. We will explain them in more detail later and will also cover them in class. During the course you will get more familiar with using `.toml` files and the variables you can specify.

#### 2. Rye automatically installs dependencies

**The dependencies of your project specified in `pyproject.toml` are automatically installed when you run `rye sync`. This is another reason why we chose rye**, as it is a big improvement over `pip` or `conda`. These require you to manually specify and update the dependencies in a `requirements.txt` file. This file is not automatically updated when you install new packages. A recipe for disaster, as it is very easy to forget to update the `requirements.txt` file and can be become extremely hard to detect conflicts.

#### 3. rye automatically handles dependencies

Rye can do this automatically, because it uses the `requirements.lock` file to store the exact versions of the packages you install and all the correct versions of the dependencies that these packages require. Most (practically all) packages require other packages to function. They all depend (on the right version) of each other (hence the name `dependency management`).

For example, this `pyproject.toml` example specifies that the `pandas` package is installed in version `2.0.3`:

```toml
dependencies = [
    "scikit-learn>=1.3.0",
    "pandas>=2.0.3", <--- this line specifies the pandas version
    "jupyter>=1.0.0",
    "numpy>=1.24.4",
    "datascience-cookiecutter>=0.3.3",
]
```

This is translated automatically to the requirements.lock file. Have a look at it right now in the root of this repository. You can see that the `pandas` package has a lot of dependencies.

As you can see, the `.lock` file specifies the exact versions of other packages that this version of pandas requires to function correctly (and the related python versions).

Rye install will handle all of this, but 10x faster than pip. The same goes for when you install or update a package. It will also automatically solve any conflicts between packages. This means that if a certain package has constraints on its dependencies, Rye will automatically install the correct versions that works with all the other packages in your project. If Rye finds a solution to this puzzle, it creates a `.lock` file that pins all versions.

Also, when you install a new package with `rye add`, it wil automatically update both `pyproject.toml` and `requirements.lock`. These file changes are then tracked by git (they need to be committed), which ensures that everybody working on the project can verify why which package (and version) was added.

You can imagine when you use a lot of packages it will become very hard (and easy to mess up) to keep track of all their dependencies. **So the third and main reason why we recommend rye, is that it helps you create deterministic environments, that are easily portable to multiple platforms.**

#### 4. Other reasons to use rye

Other reasons to use rye are that it works on Windows, Linux and macOS and that it is very easy to publish a package to [Pypi](https://pypi.org/). While this might not be something a beginner will do often, it is still helpful to have good tools that make this easy once you reach a level where you want or need to do this.

## rye in practice

If you are not used to working with virtual environments, it is a lot of conceptual information at once. In short, you can imagine that rye creates a pool of Python versions installed on your system: there is one place and one tool where all your Python versions are stored and tracked. In combination with a rye virtual environment, you enforce that your project chooses the correct Python version for it's packages from this pool and automatically keep track of all the necessary dependencies. Hereby keeping your project isolated from other projects on your system.

Using rye is relatively simple.

### if you dont have a `.toml` file yet

You can create a new virtual environment for your project by running `rye init` in your terminal, in the root directory of your project. This will create a `pyproject.toml` file in this directory. You can add packages with `rye add`.

### if you already have a `.toml` file

Often you will find yourself cloning a repository, like you have done with this repository. When you cloned it, you already pulled in a `pyproject.toml` file and a `requirements.lock` file from the origin repository. In this case, with the files already provided, you can simply run `rye sync` to install all the packages specified in `pyproject.toml` and their dependencies.

> **Note:** this is a benefit of using `pyproject.toml` and `requirements.lock` files. As files in the repository, they represent the common "truth" of all the requirements for a project or application to run correctly.

In both cases rye will create a `.venv` folder in your project root directory, with default settings.

The folder created (eg `.venv`) is where your virtual environment for your project "resides". Don't worry about this for now. Just remember that this is where your project's Python version and packages are stored, and that it is a folder like every other folder so you can look around inside it. Vscode can help you locate code you import from your environment by right-clicking and selecting `Go to Definiton`.

You activate the environment with `source .venv/bin/activate` and deactivate it with `deactivate`. For windows, use `source .venv/Scripts/activate` and `deactivate` respectively.

To familiarize yourself more with rye, we encourage you watch the intro video:
[rye](https://rye.astral.sh/guide/)