#!/bin/bash
env > /tmp/env.txt
export PATH="/home/azureuser/.cache/pypoetry/virtualenvs/deep-learning-ho7aY0_Y-py3.9/bin:/home/azureuser/.julia/juliaup/bin:/home/azureuser/.local/bin:/home/azureuser/.julia/juliaup/bin:/home/azureuser/.local/bin:/home/azureuser/.pyenv/shims:/home/azureuser/.pyenv/bin:/home/azureuser/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:$PATH"

cd /home/azureuser/code/ML22/notebooks/7_whatsnext/explainable/

make model
make shap

env > /tmp/done.txt


