install-dev: ## Install dev dependencies
	uv sync --extra dev

install-all: ## Install all dependencies
	uv sync --all-extras

notebook-docs: ## Build the markdown versions of notebook files for docs site
	IS_GENERATING_DOCS=true uv run jupyter nbconvert docs/notebooks/*.ipynb \
		--config jupyter_nbconvert_config.py \
		--NbConvertApp.output_files_dir notebook-assets \
		--to markdown

notebook-docs-debug:
	IS_GENERATING_DOCS=true uv run jupyter nbconvert docs/notebooks/*.ipynb \
		--config jupyter_nbconvert_config.py \
		--NbConvertApp.output_files_dir notebook-assets \
		--log-level=DEBUG \
		--to markdown

serve: install-dev ## Serve documentation website for local development
	uv run mkdocs serve

docs: install-dev ## Build documentation website directory
	uv run mkdocs build

lint: ## Lint source with ruff
	uv run ruff check src/

fmt: ## Format source with ruff
	uv run ruff format src/

test: ## Run unit and local-fixture tests (excludes live network tests)
	uv run pytest

test-live: ## Run only the live network tests (requires pol.is access)
	uv run pytest -m live

h5ad: ## Export a Polis conversation to h5ad (writes to exports/)
	uv run python scripts/export_h5ad.py

build: ## Build wheel package for publishing to PyPI
	rm -rf dist/
	uv build

publish: ## Publish built package to PyPI
	uv publish

# These make tasks allow the default help text to work properly.
%:
	@true

.PHONY: help notebook-docs notebook-docs-debug serve docs lint fmt test test-live h5ad build publish

help:
	@echo 'Usage: make <command>'
	@echo
	@echo 'where <command> is one of the following:'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
