.PHONY: help mnist model shap

.DEFAULT: help

help:
	@echo "make mnist"

mnist:
	make model
	make shap

mnistfashion:
	make modelfashion
	make shap

model:
	export PYTHONPATH="$$HOME/code/ML22" ; poetry run python 02_image_explainer.py

shap:
	export PYTHONPATH="$$HOME/code/ML22" ; poetry run python 03_shap_values.py

tb:
	poetry run tensorboard --logdir=../../models/embeddings

model:
	export PYTHONPATH="$$HOME/code/ML22" ; poetry run python 02_image_explainer.py