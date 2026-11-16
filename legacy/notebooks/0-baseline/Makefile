.PHONY: help install build test lint format

.DEFAULT: help
help:
	@echo "make lint"
	@echo "       run flake8 and mypy"
	@echo "make format"
	@echo "       run isort and black"
	@echo "make help"
	@echo "       print this help message"

lint:
	pdm run ruff elfarol --fix
	pdm run mypy elfarol  --ignore-missing-imports

format:
	pdm run black elfarol

