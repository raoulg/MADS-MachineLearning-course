.PHONY: help run encode viz embed query tb

.DEFAULT: help

PYTHONPATH := $(PWD)

help:
	@echo "make run"

run:
	make encode
	make viz
	make embed
	make query
	make tb

encode:
	echo $(PYTHONPATH)
	export PYTHONPATH=$(PYTHONPATH) & pdm run 01_autoencoder.py
	# export PYTHONPATH=$(PTYHONPATH) ; pdm run python 01_autoencoder.py

viz:
	export PYTHONPATH=$(PYTHONPATH) ; pdm run python 02_show_vae.py

embed:
	export PYTHONPATH=$(PYTHONPATH) ; pdm run python 03_vae_embeddings.py

query:
	export PYTHONPATH=$(PYTHONPATH) ; pdm run python 04_query_embeddings.py

tb:
	pdm run tensorboard --logdir=../../models/embeddings

