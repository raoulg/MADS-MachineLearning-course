.PHONY: help run encode viz embed query tb

.DEFAULT: help

help:
	@echo "make run"

run:
	make encode
	make viz
	make embed
	make query
	make tb

encode:
	export PYTHONPATH="$$HOME/code/ML22" ; poetry run python 01_autoencoder.py

viz:
	export PYTHONPATH="$$HOME/code/ML22" ; poetry run python 02_show_vae.py

embed:
	export PYTHONPATH="$$HOME/code/ML22" ; poetry run python 03_vae_embeddings.py

query:
	export PYTHONPATH="$$HOME/code/ML22" ; poetry run python 04_query_embeddings.py

tb:
	tensorboard --logdir=../../models/embeddings

