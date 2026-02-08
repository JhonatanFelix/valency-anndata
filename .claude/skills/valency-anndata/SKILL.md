---
name: valency-anndata
description: >
  Helper for the valency-anndata Python package — an AnnData-based toolkit for analyzing
  Polis opinion/voting data. Use when working with Polis conversations, running the Polis
  analysis pipeline (recipe_polis), loading datasets, preprocessing vote matrices, clustering
  participants, or visualizing results. Triggers on: AnnData vote matrices, Polis data,
  opinion clustering, val.datasets/val.pp/val.tl/val.viz usage, recipe_polis, schematic diagrams,
  langevitour, jupyter-scatter, or any valency-anndata API call.
---

# Valency AnnData

Toolkit for analyzing Polis opinion/voting data using AnnData. Follows scanpy namespace conventions.

## API Namespace

```
import valency_anndata as val

val.datasets    # Load Polis conversation data
val.pp          # Preprocessing (alias: val.preprocessing)
val.tl          # Analysis tools (alias: val.tools)
val.viz         # Visualization
val.scanpy      # Re-exported scanpy (pp, tl, pl, get)
```

## Data Model

Core structure: `participants x statements` AnnData matrix. Votes are `-1`/`+1` with `NaN` for unseen.

- `.X` — vote matrix
- `.obs` — participant metadata + QC metrics + cluster labels
- `.var` — statement metadata (`content`, `is_meta`, `moderation_state`, etc.)
- `.layers` — intermediate matrices (`X_masked`, `X_masked_imputed_mean`)
- `.obsm` — embeddings (`X_pca_polis`, `X_pacmap`, `X_umap`)
- `.uns` — raw votes, statements, pipeline params

For full data model details, see `references/data-model.md`.

## Loading Data

```python
# From Polis report URL
adata = val.datasets.polis.load("https://pol.is/report/r29kkytnipymd3exbynkd")

# From conversation URL
adata = val.datasets.polis.load("https://pol.is/4asymkcrjf")

# Bare IDs work too (report IDs start with 'r', conversation IDs start with digit)
adata = val.datasets.polis.load("r2dxjrdwef2ybx2w9n3ja")

# Custom host
adata = val.datasets.polis.load("https://polis.tw/report/r29kkytnipymd3exbynkd")

# Local directory (must contain votes.csv and comments.csv)
adata = val.datasets.polis.load("/path/to/export/")

# With translation
adata = val.datasets.polis.load("...", translate_to="en")

# Pre-packaged datasets
adata = val.datasets.aufstehen(translate_to="en")  # Largest Polis conversation (33k+ participants)
```

## The Polis Pipeline (recipe_polis)

End-to-end Small et al. pipeline. Run with:

```python
val.tools.recipe_polis(adata)
```

**Six sequential steps:**

1. **`_zero_mask()`** — Mask metadata & moderated statements. Requires `.var["is_meta"]`. Creates `.layers["X_masked"]`.
2. **`impute()`** — Column-mean imputation of NaN values. Creates `.layers["X_masked_imputed_mean"]`.
3. **`pca()`** — Standard PCA on imputed matrix. Creates `.obsm["X_pca_masked_unscaled"]`.
4. **`_sparsity_aware_scaling()`** — Divides PCA by sparsity scaling factors (via reddwarf). Creates `.obsm["X_pca_polis"]`.
5. **`_cluster_mask()`** — Exclude participants with < 7 votes from clustering. Creates `.obs["cluster_mask"]`.
6. **`kmeans()`** — Silhouette-scored k-means (k=2..5). Creates `.obs["kmeans_polis"]`.

**Key parameters:**

```python
val.tools.recipe_polis(
    adata,
    participant_vote_threshold=7,   # Min votes for clustering
    key_added_pca="X_pca_polis",    # PCA embedding key
    key_added_kmeans="kmeans_polis", # Cluster label key
    inplace=True,                    # Modify adata in-place
)
```

**Custom pipelines** can import helper steps directly:

```python
from valency_anndata.tools._polis import _zero_mask, _cluster_mask, _sparsity_aware_scaling
```

## Statement Clustering (recipe_polis2)

LLM-based statement clustering (requires `pip install valency-anndata[polis2]`):

```python
val.tools.recipe_polis2_statements(adata)
# Creates: .varm["content_embedding"], .varm["content_umap"], .var["evoc_polis2_top"]
```

## Preprocessing

```python
# Impute missing values (strategies: "mean", "zero", "median")
val.pp.impute(adata, strategy="mean", source_layer="X_masked", target_layer="X_masked_imputed_mean")

# QC metrics (adds n_votes, pct_agree, pct_seen, mean_vote, etc. to .obs and .var)
val.pp.calculate_qc_metrics(adata, inplace=True)

# Rebuild vote matrix from raw votes (useful for time-trimming)
val.pp.rebuild_vote_matrix(adata, trim_rule=1.0, inplace=True)

# Scanpy re-exports
val.pp.neighbors(adata, ...)
```

## Tools (Beyond recipe_polis)

```python
val.tl.kmeans(adata, ...)       # Standalone k-means
val.tl.pacmap(adata)            # PaCMAP embedding
val.tl.localmap(adata)          # LocalMAP embedding
val.tl.pca(adata, ...)          # Scanpy PCA
val.tl.umap(adata, ...)        # Scanpy UMAP
val.tl.tsne(adata, ...)         # Scanpy t-SNE
val.tl.leiden(adata, ...)       # Leiden clustering
```

## Visualization

```python
# Schematic diagram — SVG of AnnData structure
val.viz.schematic_diagram(adata)

# Context-manager mode — snapshots before/after, renders diff
with val.viz.schematic_diagram(diff_from=adata):
    val.tools.recipe_polis(adata)

# Scanpy plots
val.viz.pca(adata, color="kmeans_polis")
val.viz.embedding(adata, basis="pacmap", color=["kmeans_pacmap", "pct_seen"])

# Interactive exploration
val.viz.langevitour(adata, use_reps=["X_umap", "X_pca[:10]"], color="leiden")
val.viz.jscatter(adata, ...)
```

## Typical Notebook Workflow

```python
import valency_anndata as val

# 1. Load
adata = val.datasets.polis.load("https://pol.is/report/r29kkytnipymd3exbynkd")

# 2. Translate (optional)
val.datasets.polis.translate_statements(adata, translate_to="en")

# 3. Inspect initial structure
val.viz.schematic_diagram(adata, diff_from=None)

# 4. Run pipeline with visual diff
with val.viz.schematic_diagram(diff_from=adata):
    val.tl.recipe_polis(adata)

# 5. QC
val.pp.calculate_qc_metrics(adata, inplace=True)

# 6. Visualize
val.viz.pca(adata, color="kmeans_polis")
val.viz.embedding(adata, basis="pacmap", color=["kmeans_pacmap", "pct_seen", "pct_agree"])

# 7. Explore interactively
val.viz.langevitour(adata, use_reps=["X_umap", "X_pca[:10]"], color="leiden")
```

## Common Gotchas

- `.var["is_meta"]` **must exist** before `recipe_polis` — `ValueError` otherwise.
- `ipywidgets<8` is pinned for Colab compatibility. Don't bump without testing Colab.
- `setuptools<81` required because langevitour imports setuptools at runtime.
- PaCMAP crashes notebook kernel on Python 3.10 in CI — use 3.11+ for CI.
- Private modules use `_underscore` prefix. Only functions in `__init__.py` are public API.
- Use `uv run` for all commands (project uses uv exclusively).

## Development

```bash
uv sync --extra dev       # Install dev dependencies
uv run ruff check src/    # Lint
uv run ruff format src/   # Format
make test                 # Run tests
make test-live            # Run tests requiring network
make serve                # Serve docs locally
make docs                 # Build docs site
```
