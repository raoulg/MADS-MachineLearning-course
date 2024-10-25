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
	python src/autoencoder.py

viz:
	python src/show_vae.py

embed:
	python src/vae_embeddings.py

query:
	python src/query_embeddings.py

tb:
	tensorboard --logdir models/embeddings

lint:
	black src
	isort src
	ruff src --fix
	mypy src --pretty --ignore-missing-imports