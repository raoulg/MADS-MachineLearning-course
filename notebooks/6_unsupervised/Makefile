.PHONY: help run encode viz embed query tb

.DEFAULT: help

PYTHONPATH := $(PWD)

help:
	@echo "make all"

all:
	make encode
	make viz
	make embed
	make query
	make tb

encode:
	export PYTHONPATH=$(PYTHONPATH) & pdm run src/autoencoder.py

viz:
	export PYTHONPATH=$(PYTHONPATH) ; pdm run python src/show_vae.py

embed:
	export PYTHONPATH=$(PYTHONPATH) ; pdm run python src/vae_embeddings.py

query:
	export PYTHONPATH=$(PYTHONPATH) ; pdm run python src/query_embeddings.py

tb:
	pdm run tensorboard --logdir models/embeddings

lint:
	black src
	isort src
	ruff src --fix
	mypy src --pretty --ignore-missing-imports