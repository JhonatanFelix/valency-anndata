# Changelog

## [Unreleased][] (YYYY-MM-DD)

### Added
- `CLAUDE.md` guidance file for Claude Code contributors ([#58][]).
- Pytest infrastructure and test suite for `datasets.polis.load` ([#59][]).
  - 29 unit + local-fixture tests; 4 opt-in live network tests (`make test-live`).
  - Synthetic and real CSV fixtures checked in under `tests/fixtures/`.
  - `make test` and `make test-live` targets added to Makefile.
- Unit and integration tests for `tools.kmeans` ([#63][]).
  - 22 mocked unit tests + 1 real-clustering integration test.
  - 3 `k-means++` smoke tests.
- `val.tl.recipe_polis2_statements()` — embeds and clusters statements (var axis) via polismath ([#44][]).
  - New `polis2` optional-dependency group (`pip install valency-anndata[polis2]`).
  - 13 unit tests with all polismath helpers mocked.
  - Noise/unassigned cluster labels (`-1`) in `evoc_polis2_top` are stored as `NA` so scanpy renders them as lightgray by default.
  - `show_progress=False` (the default) now fully silences HF download progress bars and mlx model-load stdout.
  - "Polis 2.0 Pipeline" tutorial added to docs nav.
- `val.preprocessing.highly_variable_statements()` — identify highly variable statements in vote matrices ([#52][]).
  - Analogous to scanpy's highly_variable_genes for single-cell data.
  - Supports multiple variance modes (overall, valence, engagement) and binning strategies.
  - `key_added` parameter allows running multiple times with different settings.
  - `val.viz.highly_variable_statements()` plotting function for visualizing dispersion metrics.
  - `mask_var` parameter added to `val.tools.recipe_polis()`, `val.tools.pacmap()`, and `val.tools.localmap()` for filtering statements before dimensionality reduction.
- `make lint` and `make fmt` targets for ruff.
- Claude Code skill for guided Polis conversation exploration ([#42][]).
  - Interactive prompts for projection selection (PaCMAP, LocalMAP, UMAP, t-SNE) and QC annotation selection.
  - Fixed CLI plotting to support multi-color `val.viz.embedding()` calls.

[#42]: https://github.com/patcon/valency-anndata/issues/42
[#44]: https://github.com/patcon/valency-anndata/issues/44
[#52]: https://github.com/patcon/valency-anndata/issues/52
[#58]: https://github.com/patcon/valency-anndata/pull/58
[#59]: https://github.com/patcon/valency-anndata/issues/59
[#63]: https://github.com/patcon/valency-anndata/pull/63

## [0.1.1][] (2026-01-20)

### Fixes
- Fixed the README image so not broken on PyPI.

## [0.1.0][] (2026-01-19)

Initial release includes:

- `val.viz.schematic_diagram()` helper to showing data structure and visual diffs.
- Helper methods for PaCMAP and LocalMAP dimensional reduction.
- Langevitour visualisation for exploring high dimensional space.
- Basic Jupyter Scatter support for up to 1M participants, including animations.
- Import of Polis conversation data.
- Basic Polis v1 pipeline support.
- Added `val.tools.kmeans()`.
- Large reference datasets for Aufstehen political party consultation (33k participants) and the #ChileDesperto protest (3k).
- `val.viz.voter_vignette_widget()` for exploring data stories of random individuals.
- Comprehensive documentation website.
- Wrappers for various scanpy methods.

<!-- Links -->

[Unreleased]: https://github.com/patcon/valency-anndata/compare/v0.1.1...main
[0.1.0]: https://github.com/patcon/valency-anndata/releases/tag/v0.1.0
