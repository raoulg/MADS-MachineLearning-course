# Hypertuning
### data
You can pick the gestures dataset, which is pretty fast en easy to reach high accuracy on.
You could also pick the flowers if you want to test images, or the ants/bees, or any other datasets from `mads_datasets` or even another set you like from [torchvision](https://pytorch.org/vision/0.8/datasets.html) or [torchtext](https://pytorch.org/text/stable/datasets.html#imdb).

Keep in mind that your hardware will limit what you can do with the images; if you have a GPU or use a free GPU from google colab, it's much faster to run tests.

## Create a model
We have worked with a few types of layers:
- linear
- conv2d
- RNNs (GRU, LSTM)
- dropout, batchnorm

and we have also seen architectures like resnet (skiplayers), squeeze-excite and googlenet (inception).
It's up to you to create a configurable model now that can be hypertuned.

## Goal
How do variables interact?
This means:
- pick a variable for the x-axis
- pick another variable for the y-axis
now you can use a metric as a third variable for color, and create heatmaps to explore how the two variables interact.

You will typically see that variables interact. For example, dropout will exclude random hidden units from a layer. This means that if you have more units, your dropout can be higher. Some variables you can consider, ranging from high to low importance:
- the architecture of the model is the most important. Number of layers, type of layers, skiplayers, etc.
- the number of hidden units
- learning rate in combination with an optimizer
- dropout
- batchnorm

The notebook 02_hypertune is an illustration of the interaction between number of layers and hidden units.
You will pick two other variables and explore their interaction.

IMPORTANT:
Dont use hyperband when trying to create a heatmap! Or, at least, dont put models that have run just a few epochs together with models that have run many epochs. This will NOT give you a clear overview of the interaction between the variables.

You CAN use hyperband if you want to speed up scans of big hyperparameter spaces, but always keep in mind that you DONT use this for plotting heatmaps.

- Visualise your finding
- reflect on what you see, using the theory. Please dont use chatGPT to reflect: you will get text that makes you look like an idiot. Instead, try to make mistakes, it wont be subtracted from your grade; making mistakes gives me an oportunity to correct you, which might actually help you during your exam.


## Implementing
Set up your own repository.
In hypertune.py, I have set up an example for hypertuning.
Implement your own hypertuner for another model / dataset from scratch.

- make sure your environment is reproducable
    - dont blindly reuse the environment from the lessons; use a minimal environment.
    - Check the [dependencies for mltrainer](https://github.com/raoulg/mltrainer/blob/main/pyproject.toml) to see whats already installed if you use mltrainer
    - have a look at the pyproject.toml file from the course;
        - `hpbandster` is for hyperband
        - `configspace` is a dependency for `hpbandster`.
        - `bayesian-optimization` is for, well, bayesian optimization
        - `hyperopt` is another hyperoptimalisation library which we have used in `2_convolutions/03_mlflow.ipynb`. It doesnt has as much algorithms as ray, which is why we use ray. Just know that `hyperopt` can be an alternative to `ray.tune` that's slightly easier to setup but less powerful.
- make a function to pull your data, you can use `mads_datasets` for this.
- make a *configurable model* where you can change the different options from the config
- build a hypertuner.py script
    - DO NOT use hyperband!
- make a README that explains your experiments
- Lint and format your code !! with black and ruff untill all your errors are gone !!

## Report
You will write a report of 1 page. Not 2, not 3. 1 page. Somehow this turns out to be very difficult for students :)
1 A4 page means you will have to be very clear with your report. Remove all clutter. Use clear visualisations. Make sure it's clear in a few seconds what the results are of every experiment.

You will get a:
- 0, which means: you will have to improve things, otherwise it will cost you points at your exam
- 1, meaning: your work is fine. There are some things to improve, but you are on the right track.
- 2, meaning: your work is excellent. You have done everything right, even exceeded expectations,and you have shown that you understand the material. If you do the same on the exam you will be good.

# Common gotchas

- you can't blindly copy paths. Always use `Path` and check if locations exist with `.exists()`
- If you want to add your own `src` folder for import, you need to figure out if your `src` folder is located at the ".." or "../.." location, relative to your notebook for a `sys.path.insert(0, "..")` command (that is, if you need to explicitly add it because the notebook cant find your src folder)
- same goes for datalocations. `"../../data/raw"` might have changed into `"../data/raw"` depending in your setup. `mads_datasets` uses `Path.home() / ".cache/mads_datasets"` as a default location, so you can use that from every folder on your computer.
- While developing functions, you can:
    1. Write the function in a .py file, and (re)load into a notebook if you change it. Note that ray tune can be problematic to run from inside a notebook.
    2. Make a hypertune.py file and excecute from the command line. You will never have problems with reloading functions, because that is done every time you start up.
- PRACTICE linting and formating (using a Makefile makes this easier). Black is simple, ruff too (it has a --fix argument to autofix issues), mypy takes more effort (because you need to typehint) but you will become a better programmer if you learn to think about the types of your input and output. Additionally mypy will catch possible errors that dont show up during a first run (but might show up later, with different input)