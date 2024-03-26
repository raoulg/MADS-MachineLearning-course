deep learning course for Hogeschool Utrecht
==============================

This course is the last course in a series of modules for data science.
This course assumes you have done the introduction in Python and the DME course https://github.com/raoulg/MADS-DAV

For this project you will need some dependencies.
The project uses python 3.10, and all dependencies can be installed with `pdm` if you are using that with `pdm install`.

If you prefer to use `pip` you can run `pip install -r requirements.txt`.

The lessons can be found inside the `notebooks`folder.
The source code for the lessons can be found in the `src`folder.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make format` or `make lint`
    ├── README.md          <- The top-level README for developers using this project.
    ├── .gitignore         <- Stuff not to add to git
    ├── poetry.lock        <- Computer readable file that manages the dependencies of the libraries
    ├── pyproject.toml     <- Human readable file. This specifies the libraries I installed to
    |                         let the code run, and their versions.
    ├── requirements.txt   <- Export from poetry.lock to requirements.txt, to be used with pip
    ├── setug.cfg          <- Config file for linters etc.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
