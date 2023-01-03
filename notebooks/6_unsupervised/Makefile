.PHONY: help run encode viz embed tb

.DEFAULT: help

help:
	@echo "make run"

run:
	make encode
	make viz
	make embed
	make tb

encode:
	export PYTHONPATH="$(shell realpath ../..)" ; poetry run python 01_autoencoder.py

viz:
	export PYTHONPATH="$(shell realpath ../..)" ; poetry run python 02_show_vae.py

embed:
	export PYTHONPATH="$(shell realpath ../..)" ; poetry run python 03_vae_embeddings.py

tb:
	tensorboard --logdir=../../models/embeddings

