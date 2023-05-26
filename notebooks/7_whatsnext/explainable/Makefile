.PHONY: help mnist mnistfashion model modelfashion shap shapfasion

.DEFAULT: help
PYTHONPATH := /workspaces/ML22

help:
	@echo "make mnist"
	@echo "make mnistfashion"

mnist:
	make model
	make shap
# open img/final.jpeg
	make tb

mnistfashion:
	make modelfashion
	make shapfashion
# open img/final.jpeg
	make tb

model:
	export PYTHONPATH=$(PYTHONPATH) ; poetry run python 02_image_explainer.py mnist

modelfashion:
	export PYTHONPATH=$(PYTHONPATH) ; poetry run python 02_image_explainer.py mnistfashion

shap:
	export PYTHONPATH=$(PYTHONPATH) ; poetry run python 03_shap_values.py mnist

shapfashion:
	export PYTHONPATH=$(PYTHONPATH) ; poetry run python 03_shap_values.py mnistfashion

tb:
	poetry run tensorboard --logdir=../../../models/explainer
