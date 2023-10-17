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
	poetry run mypy --config-file tox.ini elfarol

format:
	poetry run isort -v elfarol
	poetry run black elfarol
