import numpy as np
import pandas as pd
from anndata import AnnData


def _embed_statements(texts: list[str]) -> np.ndarray:
    """Embed statement texts using polismath_commentgraph.

    Returns
    -------
    np.ndarray
        Shape (n_statements, embed_dim).
    """
    try:
        from polismath_commentgraph.core import EmbeddingEngine
    except ImportError as exc:
        raise ImportError(
            "polismath-commentgraph is required for polis2 recipes. "
            "Install it with: pip install valency-anndata[polis2]"
        ) from exc

    return EmbeddingEngine().embed_batch(texts=texts, show_progress=True)


def _project_umap(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2-D via UMAP using polismath_commentgraph.

    Returns
    -------
    np.ndarray
        Shape (n_statements, 2).
    """
    try:
        from polismath_commentgraph.core import ClusteringEngine
    except ImportError as exc:
        raise ImportError(
            "polismath-commentgraph is required for polis2 recipes. "
            "Install it with: pip install valency-anndata[polis2]"
        ) from exc

    return ClusteringEngine().project_to_2d(embeddings=embeddings)


def _create_cluster_layers(embeddings: np.ndarray, num_layers: int = 4) -> list[np.ndarray]:
    """Create hierarchical clustering layers using polismath_commentgraph.

    Returns
    -------
    list[np.ndarray]
        List of length *num_layers*.  ``layers[0]`` is the finest granularity;
        ``layers[-1]`` is the coarsest.  ``-1`` indicates noise / unassigned.
    """
    try:
        from polismath_commentgraph.core import ClusteringEngine
    except ImportError as exc:
        raise ImportError(
            "polismath-commentgraph is required for polis2 recipes. "
            "Install it with: pip install valency-anndata[polis2]"
        ) from exc

    return ClusteringEngine().create_clustering_layers(
        embeddings=embeddings, num_layers=num_layers
    )


def recipe_polis2_statements(adata: AnnData, *, inplace: bool = True) -> AnnData | None:
    """Embed and cluster **statements** (the var axis) using the Polis v2 pipeline.

    Reads free-text statement content from ``.var["content"]``, produces
    dense embeddings, projects them to 2-D with UMAP, and attaches a
    hierarchy of cluster labels — all stored on the **var** axis so that
    the results live alongside the statements that produced them.

    Requires the optional ``polis2`` dependency group::

        pip install valency-anndata[polis2]

    Recipe Steps
    ------------

    1. Embeds each statement's text into a high-dimensional vector space
       and stores the result in ``.varm["X_text_embed"]``.
    2. Projects the embeddings to 2-D with UMAP and stores the coordinates
       in ``.varm["X_umap_statements"]``.
    3. Builds a hierarchy of clustering layers (finest → coarsest) and
       stores them in ``.varm["evoc_polis2"]`` (shape ``n_var × num_layers``)
       with the coarsest layer also surfaced as the categorical column
       ``.var["evoc_polis2_top"]``.

    Parameters
    ----------
    adata :
        AnnData object whose ``.var["content"]`` column contains the
        statement text strings.
    inplace :
        If ``True`` (default), mutate *adata* and return ``None``.
        If ``False``, operate on a copy and return it.

    Returns
    -------
    Depending on *inplace*, returns ``None`` or the modified ``AnnData``.

    .varm['X_text_embed']
        Dense text embeddings, shape ``(n_var, embed_dim)``.
    .varm['X_umap_statements']
        2-D UMAP projection of the embeddings, shape ``(n_var, 2)``.
    .varm['evoc_polis2']
        Stacked clustering layers, shape ``(n_var, num_layers)``.
        Column 0 is the finest; column -1 is the coarsest.  ``-1`` = noise.
    .var['evoc_polis2_top']
        Categorical column taken from the coarsest clustering layer
        (i.e. ``evoc_polis2[:, -1]``).

    Examples
    --------
    >>> import valency_anndata as val
    >>> adata = val.datasets.polis.load("https://pol.is/report/r3nubzxvjara8ccesdsau")  # doctest: +SKIP
    >>> val.tl.recipe_polis2_statements(adata)  # doctest: +SKIP
    >>>
    >>> # Transpose so statements sit on the obs axis for plotting.
    >>> adata_t = adata.transpose()  # doctest: +SKIP
    >>> # Extract a UMAP dimension into obs for use as a colour channel
    >>> # (workaround until #55 lands):
    >>> adata_t.obs["umap_1"] = adata_t.obsm["X_umap_statements"][:, 0]  # doctest: +SKIP
    >>> val.viz.embedding(adata_t, basis="umap_statements", color=["evoc_polis2_top", "umap_1"])  # doctest: +SKIP
    """
    if not inplace:
        adata = adata.copy()

    texts = adata.var["content"].tolist()

    adata.varm["X_text_embed"] = _embed_statements(texts)
    adata.varm["X_umap_statements"] = _project_umap(adata.varm["X_text_embed"])

    cluster_layers = _create_cluster_layers(adata.varm["X_text_embed"])
    adata.varm["evoc_polis2"] = np.array(cluster_layers).T
    adata.var["evoc_polis2_top"] = pd.Categorical(adata.varm["evoc_polis2"][:, -1])

    if not inplace:
        return adata
