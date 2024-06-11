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

# Barebones Automation

A typical situation would be that you want to execute your pipeline every tuesday at 15:00.
Let's say you have a VM that starts up at 1430 automatically, and now you want a script to kick off.
The most reliable way to do this, is with a shell script

## shell script
You create shell script with the following commands. The chmod command adds execution rights to the .sh script, which is why you need sudo.

```bash
> touch run.sh
> chmod +x run.sh
```

First, we are going to fill the `run.sh` file with a line that tells the script to use bash, and a line to print your environment. Why would you want to do the latter? Because crontab environment are typically a painfull source of hours of debugging :)

Copy the lines below into your `run.sh` file:
```bash
#!/bin/bash
env > /tmp/env1.txt
```

Crontab will have a bare bones shell environment. But, for our setup to work, the script will need to know where it can find things like:
- the poetry command
- our virtual environment
- our python installation

The quick and dirty way to do this, is to go to your project folder and to activate your environment with `poetry shell`. This will add the necessary stuff to your PATH. After that you simply print your full PATH variable to the screen with `echo`:

```bash
> poetry shell
> echo $PATH
```
First,

Now you can copy-paste the output of that into this line of code in your .sh script:

```bash
export PATH="<output from echo>:$PATH"
```

Your `run.sh` file should look a bit like this:
```bash
#!/bin/bash
env > /tmp/env1.txt
export PATH="/home/azureuser/.cache/pypoetry/virtualenvs/deep-learning-ho7aY0_Y-py3.9/bin:/home/azureuser/.julia/juliaup/bin:/home/azureuser/.local/bin:/home/azureuser/.julia/juliaup/bin:/home/azureuser/.local/bin:/home/azureuser/.pyenv/shims:/home/azureuser/.pyenv/bin:/home/azureuser/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:$PATH"
```

We have a lot of stuff in PATH we dont need, but that's not going to hurt anyone.

Now we will add the commands we want our script to execute:

```bash
cd /home/azureuser/code/ML22/notebooks/6_unsupervised/
make encode
make viz
make embed
make query
env > /tmp/done.txt
```

We `cd` into the right directory, we run our `make` commands, and finally print a `done.txt` file to the `tmp` folder just to help us.

## crontab
Now we want this script to be executed at a certain moment in time. We will use crontab for this:

```bash
> sudo crontab -e
```
For more documentation on how to work with crontab, read [this](https://www.adminschoice.com/crontab-quick-reference)

We can add a rule like this to crontab:

```
*/10 * * * * /home/azureuser/run.sh
```

this would kick off the script every 10 minutes
I added this line to crontab:

```
0 14 * * 2 /home/azureuser/run.sh
```
Which means: every second day of the week (tuesday) kick off the `run.sh` script at 14:00h.

Often your linux clock will be off by an hour or so, to check this run
```bash
> date
```
in your shell. It turns out that 14:00 is 15:00 on our clock, so the crontab will kick off during the course.






