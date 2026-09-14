.PHONY: help install install-dev test lint format typecheck report clean build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install with every optional dependency and the git hooks
	pip install -e '.[all]'
	pre-commit install

test:  ## Run the test suite
	pytest

test-cov:  ## Run tests with a coverage report
	pytest --cov --cov-report=term-missing --cov-report=xml

lint:  ## Check formatting and lint rules
	ruff check src tests
	ruff format --check src tests

format:  ## Apply formatting and autofixes
	ruff format src tests
	ruff check --fix src tests

typecheck:  ## Static type check
	mypy src/deepseanet

report:  ## Summarise the committed training runs
	python -m deepseanet.cli report --results results

weights:  ## Download the released checkpoints and verify their checksums
	bash scripts/fetch_weights.sh
	python -m deepseanet.cli verify --manifest results/checkpoints.json --root .

build:  ## Build the wheel and sdist
	python -m build

clean:  ## Remove build and cache artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
