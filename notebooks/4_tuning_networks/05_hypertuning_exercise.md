# Hypertuning
### data
Pick an image dataset.
You could pick the flowers, or the ants/bees from notebook [03_minimal_transfer_learning.ipynb](03_minimal_transfer_learning.ipynb), or even another set you like from [torchvision](https://pytorch.org/vision/0.8/datasets.html) or [torchtext](https://pytorch.org/text/stable/datasets.html#imdb).

Keep in mind that your hardware will limit what you can do with the images; so you will either need to settle for smaller images, or preprocess them into a smaller size, or use google colab if the VM is too slow or limited in memory.

## Create a model
We have worked with a few types of layers:
- linear
- conv2d
- dropout
- batchnorm

and we have also seen architectures build with these layers like resnet (skip-layers), squeeze-excite and googlenet (inception).
It's up to you to create a configurable model now that can be hypertuned.

## Goal
How do variables interact?
This means:
- identify a pair of hyperparameters that you hypothesize will interact, based on the theory
- make plots to visualize the relationship between the hyperparameters (see lesson 5 from the [DAV](https://github.com/raoulg/MADS-DAV) course)
- pick a metric to color the plot, so you can see how the hyperparameters impact the metric

For example, from theory you would know that dropout excludes random hidden units from a layer. It is also intended to counter overfitting caused by models having too much units, causing it to be too complex. This theory gives you all sorts of predictions: for example, you would expect that if you have more units, your dropout can be higher. And that if you increase the amount of units, it will overfit, unless you increase the dropout.

Some hyperparameters to can consider, ranging from high to low importance:
- the architecture of the model is the most important. Number of layers, type of layers, skiplayers, etc.
- the number of hidden units
- dropout, batchnorm, skiplayers
- learning rate, type of optimizer, type of activation function, etc.

!!!IMPORTANT!!!:
Dont use hyperband when trying to create a heatmap! Because this will cause you to put models that have run just a few epochs together with models that have run many epochs. This will NOT give you a clear overview of the interaction between the variables.

You CAN use hyperband in the process (eg if you want to speed up scans of big hyperparameter spaces), but always keep in mind what you are doing when creating visualizations.

- Visualise your finding
- reflect on what you see, using the theory. Please dont use chatGPT to reflect: you will get text that makes you look like an idiot. Instead, try to make mistakes, it wont be subtracted from your grade; making mistakes gives me an oportunity to correct you, which might actually help you during your exam.

### Scientific method
The science part of data science is about setting up experiments. You follow this cycle:
- make a hypothesis
- design an experiment
- run the experiment
- analyze the results and draw conclusions
- repeat

To keep track of this process, it is useful to keep a journal. While you could use anything to do so, a nice command line tool is [jrnl](https://jrnl.sh/en/stable/). This gives you the advantage of staying in the terminal, just type down your ideas during the process, and you can always look back at what you have done.
Try to first formulate a hypothesis, and then design an experiment to test it. This will help you to stay focused on the goal, and not get lost in the data.

Important: the report you write is NOT the same as your journal! The journal will help you to keep track of your process, and later write down a reflection on what you have done where you draw conclusion, reflecting back on the theory.


## Implementing
In hypertune.py, I have set up an example for hypertuning.
Implement your own hypertuner for another model / dataset from scratch.

- make sure your environment is reproducable
    - dont blindly reuse the environment from the lessons; this means, DO NOT simply copy pyproject.toml but create a minimal environment by following [the codestyle](https://github.com/raoulg/codestyle/blob/main/docs/make_a_module.md)
    - Check the [dependencies for mltrainer](https://github.com/raoulg/mltrainer/blob/main/pyproject.toml) to see whats already installed if you use mltrainer
    - have a look at the pyproject.toml file from the course;
        - `configspace` is a dependency for `hpbandster`.
        - `bayesian-optimization` is for, well, bayesian optimization
        - `hyperopt` is another hyperoptimalisation library which we have used in `2_convolutions/03_mlflow.ipynb`. It doesnt has as much algorithms as ray, which is why we use ray. Just know that `hyperopt` can be an alternative to `ray.tune` that's slightly easier to setup but less powerful.
- make a function to pull your data, you can use `mads_datasets` for this.
- make a *configurable model* where you can change the different options from the config. You can decide for yourself if you want to make it configurable by simply passing a dictionary, or by creating a @dataclass as a Settings object, or by using a `.toml` file with `tomllib` (since python 3.11 part of python) and the `TOMLserializer`.
- build a `hypertuner.py` script
- make a README that explains your experiments
- Lint and format your code !! use ruff and mypy until all your errors are gone !!

## Report
Write a report of maximum 2 a4 pages. This means you will have to be very clear with your report. Remove all clutter. Use clear visualisations. Make sure it's clear in a few seconds what the results are of every experiment.

You will get a:
- 0, which means: you will have to improve things, otherwise it will cost you points at your exam
- 1, meaning: your work is fine. There are some things to improve, but you are on the right track.
- 2, meaning: your work is excellent. You have done everything right, even exceeded expectations,and you have shown that you understand the material. If you do the same on the exam you will be good.

# Common gotchas
- you can't blindly copy paths. Always use `Path` and check if locations exist with `.exists()`
- If you want to add your own `src` folder for import, the best way is to properly create your environment with `uv`. Another option is to figure out if your `src` folder is located at the ".." or "../.." location, relative to your notebook, and use a `sys.path.insert(0, "..")` command (that is, if you need to explicitly add it because the notebook cant find your src folder)
- same goes for datalocations. `"../../data/raw"` might have changed into `"../data/raw"` depending in your setup. `mads_datasets` uses `Path.home() / ".cache/mads_datasets"` as a default location, so you can use that from every folder on your computer.
- While developing functions, you can:
    1. Write the function in a .py file, and (re)load into a notebook if you change it. Note that ray tune can be problematic to run from inside a notebook.
    2. Make a hypertune.py file and excecute from the command line. You will never have problems with reloading functions, because that is done every time you start up.
- PRACTICE linting and formating (using a Makefile makes this easier). Black is simple, ruff too (it has a --fix argument to autofix issues), mypy takes more effort (because you need to typehint) but you will become a better programmer if you learn to think about the types of your input and output. Additionally mypy will catch possible errors that dont show up during a first run (but might show up later, with different input)