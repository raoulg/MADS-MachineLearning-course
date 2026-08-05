# Autoencoders

So far, we have seen supervised methods that use a label.
However, sometimes you want something more generic than you can create with a label.

One approach are autoencoders. These architectures can be used for various purposes:

- noise reduction / denoising
- image coloring
- dimensionality reduction
- watermark removal
- anomaly detection

So, as you can see, a lot of different approaches.

The architecture should also look familiar; it uses a bottleneck, just like we have seen before in the SENet implementation. An autoencoder looks like this:

<img src="https://uvadlc-notebooks.readthedocs.io/en/latest/_images/autoencoder_visualization.svg">

The novelty here is: you feed the model *the same* image that went in, as a label!

Or, with some slight variations:
- you feed the model the image + random noise as input, and the clean image as output (to learn the model to denoise images)'
- a black-and-white version as input, and a colored version as output (to learn the model to color images)
- a watermarked image as input, and a clean version as output

etc.

The bottleneck has just $z$ dimensions, where $z$ should be a low number compared to the original dimensionality.

How low this is, depends on the task at hand: if you want to detect anomalies, you definitely want $z$ to be very low, such that the model can not learn any of the anomalies (because there is only room for the most essential features).

If you want to add color, you dont want to remove so much information from the image that it is impossible for the model to reconstruct the original image, etc.

The low latent dimensionality means the model is forced the throw way anything that is not relevant, and it has to learn how to reconstruct a complete new image, just from a small vector.

# Running experiments
You should be able to run all experiments with `make run`:
- go to your shell
- use `cd` or `z` to go into the directory where you can find the `Makefile`
- run the `make all` command from the shell

This will run:
- `make encode` : this learns the embeddings.
- `make viz` to create some visualisations
- `make embed` to add some embeddings to tensorboard so you can easily visualize them
- `make query` for an example to find the 9 neighbors of a random image.
- `make tb` to fire up tensorboard

The `Makefile` consists just of commands you would otherwise run from the commandline. You can read more
about them [here](https://opensource.com/article/18/8/what-how-makefile)





