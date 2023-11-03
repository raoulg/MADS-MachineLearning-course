# Cookiecutters

## Motivation

The text below is copied from [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/) and is an excellent introduction in datascience cookiecutters:

When we think about data analysis, we often think just about the resulting reports, insights, or visualizations. While these end products are generally the main event, it's easy to focus on making the products look nice and ignore the quality of the code that generates them. Because these end products are created programmatically, code quality is still important! And we're not talking about bikeshedding the indentation aesthetics or pedantic formatting standards — ultimately, data science code quality is about correctness and reproducibility.

It's no secret that good analyses are often the result of very scattershot and serendipitous explorations. Tentative experiments and rapidly testing approaches that might not work out are all part of the process for getting to the good stuff, and there is no magic bullet to turn data exploration into a simple, linear progression.

That being said, once started it is not a process that lends itself to thinking carefully about the structure of your code or project layout, so it's best to start with a clean, logical structure and stick to it throughout. We think it's a pretty big win all around to use a fairly standardized setup like this one. Here's why:

> **Other people will thank you**
> *Nobody sits around before creating a new Rails project to figure out where they want to put their views; they just run `rails new` to get a standard project skeleton like everybody else.*

A well-defined, standard project structure means that a newcomer can begin to understand an analysis without digging in to extensive documentation. It also means that they don't necessarily have to read 100% of the code before knowing where to look for very specific things.

## Using a cookiecutter

The mentioned cookiecutter-data-science project provides a nice cookiecutter.
However, it is also very extensive, have a look at their website if you are interested.
For example, they add libraries for generating documentation, and some boilerplate code for
passing on arguments with the `click` library. While this are nice additions, for our lessons it is
typically not needed.

Thats why I created my own version of a cookiecutter.
You can find it on pypi [here](https://pypi.org/project/datascience-cookiecutter/).
This means you can easily install it with `pdm add datascience-cookiecutter` (or, of course,
`pip install datascience-cookiecutter` if you want to).

There is a default template that looks like this:

```markdown
.
├── Makefile         <- Makefile for project automation
├── README.md        <- Project documentation and instructions
├── pyproject.toml   <- Configuration file for dependencies and project metadata
├── data             <- Folder to store data
│   ├── final        <- Folder for final processed data
│   ├── processed    <- Folder for intermediate processed data
│   ├── raw          <- Folder for raw data
│   └── sim          <- Folder for simulated data
├── dev              <- Folder for development-related files
│   ├── notebooks    <- Folder for Jupyter notebooks
│   └── scripts      <- Folder for development scripts
├── docs             <- Folder for project documentation
├── myproject        <- Placeholder folder for the project itself (replaced with your project name)
│   ├── __init__.py  <- Python package initialization file
│   └── main.py      <- Main Python script for the project
├── references       <- Folder for reference materials
├── reports          <- Folder for project reports
│   ├── img          <- Folder for images and visualizations used in reports
│   └── report.md    <- Sample report file (Markdown format)
└── tests            <- Folder for project tests
```

If you dont like parts of it, you can customize the template, read the documentation if you want to know how.

In my mind, it is not necessarily about following a structure, rigidly.
However, especially for beginners, something like starting with a data directory and splitting that into raw, processed and final data is a good idea because it helps with maintaining a clean data, reproducible pipeline.
