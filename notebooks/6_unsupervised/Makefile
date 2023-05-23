.PHONY: help run encode viz embed query tb

.DEFAULT: help

PYTHONPATH := "/workspaces/ML22"

help:
	@echo "make run"

run:
	make encode
	make viz
	make embed
	make query
	make tb

encode:
	export PYTHONPATH=$(PTYHONPATH) ; poetry run python 01_autoencoder.py

viz:
	export PYTHONPATH=$(PYTHONPATH) ; poetry run python 02_show_vae.py

embed:
	export PYTHONPATH=$(PYTHONPATH) ; poetry run python 03_vae_embeddings.py

query:
	export PYTHONPATH=$(PYTHONPATH) ; poetry run python 04_query_embeddings.py

tb:
	poetry run tensorboard --logdir=../../models/embeddings

