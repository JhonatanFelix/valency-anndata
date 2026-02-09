# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Manager & Environment

This project uses **uv** exclusively. All commands should be prefixed with `uv run` to execute within the managed environment. Install dev dependencies with:

```sh
uv sync --extra dev
```

## Common Commands

| Task | Command |
|------|---------|
| Lint | `uv run ruff check src/` |
| Format | `uv run ruff format src/` |
| Run a script | `uv run python <script.py>` |
| Serve docs locally | `make serve` |
| Build docs site | `make docs` |
| Convert notebook docs | `make notebook-docs` |
| Build wheel | `make build` |
| Publish to PyPI | `make publish` |
| Test | `make test` |
| Test (live/network) | `make test-live` |

> **Always use `make` targets** (e.g. `make test`) instead of calling the underlying commands directly (e.g. `uv run pytest`). The Makefile is the source of truth for how tasks are run.

## Pull Requests & Issues

When working on a branch that references an issue number (e.g. `59-add-tests`), always include a `Closes #<number>` line at the top of the PR body so GitHub auto-closes the issue on merge.

Always add a `CHANGELOG.md` entry under the `[Unreleased]` section when making user-facing changes. Follow the existing format — group entries under `### Added`, `### Fixes`, etc. and link issue/PR numbers at the bottom of the section.

### Labels

Apply these to issues and PRs as appropriate. Descriptions are synced to GitHub via the API.

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `documentation` | Improvements or additions to documentation |
| `duplicate` | This issue or pull request already exists |
| `enhancement` | New feature or request |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention is needed |
| `invalid` | This doesn't seem right |
| `question` | Further information is requested |
| `wontfix` | This will not be worked on |
| `DevX` | Developer experience — testing, tooling, CI |
| `type:datasets` | val.datasets — data loading |
| `type:preprocessing` | val.preprocessing — preparation for analysis |
| `type:tools` | val.tools — pipeline & analysis |
| `type:viz` | val.viz — visualizations |

## Architecture Overview

The package is installed from `src/valency_anndata/` (src layout, built with hatchling). It is modelled on **scanpy**'s namespace conventions, so the top-level API mirrors scanpy's `pp` / `tl` / `pl` pattern:

```
val.datasets    — load opinion/vote data into AnnData objects
val.pp          — alias for val.preprocessing (imputation, QC, vote-matrix rebuild)
val.tl          — alias for val.tools (PaCMAP, k-means, the full Polis pipeline recipe)
val.viz         — interactive visualizations (schematic diagrams, langevitour, jupyter-scatter)
val.scanpy      — thin re-export of the underlying scanpy namespaces (pp, tl, pl, get)
```

### Data model

The core data structure is an **AnnData** object (`adata`), a `participants × statements` vote matrix. Votes are binary-valenced (`-1` / `+1`) with `NaN` for unseen statements. Layers in the AnnData are used to store intermediate matrices (e.g. `X_masked`, `X_masked_imputed_mean`). Embedding coordinates land in `.obsm` (e.g. `X_pca_polis`, `X_pacmap`). Cluster labels land in `.obs` (e.g. `kmeans_polis`).

### The Polis pipeline (`val.tools.recipe_polis`)

This is the main end-to-end workflow. It follows the red-dwarf / Small et al. pipeline: zero-mask metadata statements → impute → PCA (sparsity-aware scaling) → k-means clustering. Helper steps like `_zero_mask()` and `_cluster_mask()` are importable from `valency_anndata.tools._polis` for building custom pipelines.

### Schematic diagram (`val.viz.schematic_diagram`)

A key debugging/exploration tool. It can be used in two modes:
- **Render mode** — called with an `adata`, it renders an SVG showing the current structure.
- **Context-manager mode** — wraps a block of pipeline code; it snapshots `adata` before and after, then renders a visual diff. This is how pipeline exploration is typically done in notebooks.

### Scanpy wrappers

`tools/__init__.py` and `preprocessing/__init__.py` directly re-export scanpy functions (`pca`, `umap`, `tsne`, `leiden`, `neighbors`). These are surfaced so users never need to `import scanpy` themselves. New scanpy wrappers should follow this same pattern.

## Key Constraints & Gotchas

- **`ipywidgets<8`** is pinned to keep jupyter-scatter compatible with Google Colab. Do not bump without testing Colab.
- **`setuptools<81`** is required because langevitour expects setuptools at import time.
- **`langevitour`** is currently pulled from a personal fork (`patcon/langevitour`, rev `fix-js-indent`) via `[tool.uv.sources]`. This is only needed for the docs website build.
- **Python 3.10** is the local `.python-version`; CI uses 3.11 because pacmap crashes the notebook kernel on 3.10 in GitHub Actions.
- Private/internal modules use the `_underscore` prefix convention (e.g. `_polis.py`, `_kmeans.py`). Only functions exported in the module's `__init__.py` are part of the public API.

## Documentation

Docs are built with **MkDocs + Material theme**. API reference pages use `mkdocstrings` with numpy-style docstrings. Notebook-based tutorial pages are generated via `nbconvert` (see `jupyter_nbconvert_config.py` and `make notebook-docs`); the generated markdown lands in `docs/notebooks-autogenerated/` (gitignored). The custom preprocessor `valency_anndata.nbconvert.preprocessors.MkdocsAnnotationPreprocessor` post-processes notebook output for MkDocs compatibility.
