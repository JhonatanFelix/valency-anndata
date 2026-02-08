# AnnData Data Model Reference

## Table of Contents

- [Matrix and Axes](#matrix-and-axes)
- [Layers](#layers)
- [Observation Metadata (.obs)](#observation-metadata-obs)
- [Variable Metadata (.var)](#variable-metadata-var)
- [Embeddings (.obsm)](#embeddings-obsm)
- [Variable Embeddings (.varm)](#variable-embeddings-varm)
- [Unstructured Metadata (.uns)](#unstructured-metadata-uns)

## Matrix and Axes

```
AnnData shape: (n_participants, n_statements)
.X values: {-1, +1, NaN}
  -1 = disagree
  +1 = agree
  NaN = not seen / not voted
  0  = pass (only in raw vote events)
```

Participant IDs (`.obs_names`) and statement IDs (`.var_names`) are stored as strings.

## Layers

| Layer | Shape | Created By | Description |
|-------|-------|-----------|-------------|
| `raw_sparse` | (p, s) | `polis.load()` | Copy of original `.X` |
| `X_masked` | (p, s) | `_zero_mask()` | X with meta/moderated statement columns zeroed |
| `X_masked_imputed_mean` | (p, s) | `impute()` | Masked matrix with NaN filled by column mean |

## Observation Metadata (.obs)

### From data loading

Participant index only (voter IDs as strings).

### From `calculate_qc_metrics()`

| Column | Type | Description |
|--------|------|-------------|
| `n_votes` | int | Count of non-NaN votes |
| `n_agree` | int | Count of +1 votes |
| `n_disagree` | int | Count of -1 votes |
| `n_pass` | int | Count of 0 votes |
| `n_engaged` | int | agree + disagree |
| `pct_agree` | float | n_agree / n_votes |
| `pct_disagree` | float | n_disagree / n_votes |
| `pct_pass` | float | n_pass / n_votes |
| `pct_engaged` | float | n_engaged / n_votes |
| `pct_agree_engaged` | float | n_agree / n_engaged |
| `pct_seen` | float | n_votes / n_statements |
| `mean_vote` | float | Mean vote value (-1 to +1) |

### From recipe_polis

| Column | Type | Description |
|--------|------|-------------|
| `cluster_mask` | bool | True if participant has >= threshold votes |
| `kmeans_polis` | categorical | Cluster label (NaN if masked out) |

### From scanpy tools

| Column | Type | Description |
|--------|------|-------------|
| `leiden` | categorical | Leiden community detection labels |

## Variable Metadata (.var)

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `content` | str | load/translate | Statement text |
| `participant_id_authored` | str | load | Author participant ID |
| `created_date` | datetime | load | Statement creation timestamp |
| `is_seed` | bool | load | Whether statement was seeded by moderator |
| `is_meta` | bool | load | Whether statement is metadata (required for recipe_polis) |
| `moderation_state` | int | load | -1=moderated out, 0=active, 1+=other |
| `language_original` | str | load | Original language code |
| `language_current` | str | translate | Current language code |
| `is_translated` | bool | translate | Whether content was translated |
| `zero_mask` | bool | _zero_mask() | True if statement passes mask (~is_meta & moderation>-1) |
| `evoc_polis2_top` | categorical | recipe_polis2 | Statement cluster label (top layer) |

## Embeddings (.obsm)

| Key | Shape | Created By | Description |
|-----|-------|-----------|-------------|
| `X_pca_masked_unscaled` | (p, n_pcs) | recipe_polis step 3 | Raw PCA on imputed matrix |
| `X_pca_polis` | (p, n_pcs) | recipe_polis step 4 | Sparsity-scaled PCA (default rep) |
| `X_pacmap` | (p, 2) | `val.tools.pacmap()` | PaCMAP 2D embedding |
| `X_localmap` | (p, 2) | `val.tools.localmap()` | LocalMAP 2D embedding |
| `X_umap` | (p, 2) | `val.tools.umap()` | UMAP 2D embedding |

## Variable Embeddings (.varm)

| Key | Shape | Created By | Description |
|-----|-------|-----------|-------------|
| `X_pca_masked_unscaled` | (s, n_pcs) | recipe_polis step 3 | PCA loadings |
| `content_embedding` | (s, d) | recipe_polis2 | LLM statement embeddings |
| `content_umap` | (s, 2) | recipe_polis2 | 2D UMAP of statement embeddings |
| `evoc_polis2` | (s, n_layers) | recipe_polis2 | Hierarchical cluster assignments |

## Unstructured Metadata (.uns)

| Key | Type | Source | Description |
|-----|------|--------|-------------|
| `votes` | DataFrame | load | Raw vote events (voter_id, comment_id, vote, timestamp) |
| `votes_meta` | dict | load | Source info (via, conversation_id, report_id, retrieved_at) |
| `statements` | DataFrame | load | Statement rows from API/export |
| `statements_meta` | dict | load | Statement source metadata |
| `source` | dict | load | Basic source info |
| `schema` | dict | load | Description of X matrix and votes format |
| `X_pca_polis` | dict | recipe_polis | PCA params (variance_ratio, eigenvalues) |
| `kmeans_polis` | dict | recipe_polis | Clustering params (best_k, best_score, k_bounds, init) |
| `neighbors` | dict | scanpy | Neighbor graph parameters |
| `umap` | dict | scanpy | UMAP parameters |
| `leiden` | dict | scanpy | Leiden parameters |
