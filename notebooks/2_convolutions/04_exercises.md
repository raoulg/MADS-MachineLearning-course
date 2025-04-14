
Last week, you have been experimenting with the interaction of hyperparameters.
You made visualisations to show how they impacted each other.
This week, you will extend the number of hyperparameters and architectures you can experiment with.

# 1. Study more layers to add
Study the pytorch documentation for:
- Dropout https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- normalization layers https://pytorch.org/docs/stable/nn.html#normalization-layers

# 2. Add dropout and normalization layers to your model
Experiment with adding dropout and normalization layers to your model. Some rough guidelines where to add them relative to Linear or Conv2d layers:
- Dropout: after Linear or Conv2d layers. Often added after the last Linear layer *before* the output layer, but could occur more often.
- Normalization layers: right after (blocks of) Linear or Conv2d layers, but before activation functions.

# 3. Use logging
- set up logging with MLflow, and make sure the hyperparameters you are using are logged.
- get comfortable with using MLflow to visualize your results, it has a pretty powerful dashboard.

# 4. Adding convolutional and pooling layers
This lesson, we have added some new types of layers: convolutional and pooling layers.
Experiment with adding these new layers.

Also, have a look at the `ModuleList`: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#modulelist
This makes it much easier to use the number of layers as a hyperparameter.
You can create a list of layers from a config, and then use that list to create your model.
Instead of just adding a single layer by hand, you could also add a block of layers as a unit (eg a Conv2d layer, followed by a ReLU layer, followed by a BatchNorm2d layer, followed by a MaxPool2d layer) and repeat that in a loop, adding it to the `ModuleList`.

# 5. Reflect
- doing a master means you don't just start engineering a pipeline, but you need to reflect. Why do you see the results you see? What does this mean, considering the theory? Write down lessons learned and reflections, based on experimental results.

# 6. Make a short report
Make a short 1 a4 page report of your findings.
pay attention to:
- what was your hypothesis about interaction between hyperparameters?
- what did you find?
- visualise your results about the relationship between hyperparameters.
