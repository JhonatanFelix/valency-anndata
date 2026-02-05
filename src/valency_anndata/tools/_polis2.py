import logging
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from anndata import AnnData

_NOISY_LOGGERS = [
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
]


@contextmanager
def _quiet():
    """Suppress warnings and chatty library loggers during model loading."""
    saved = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            for name, level in saved.items():
                logging.getLogger(name).setLevel(level)


def _embed_statements(texts: list[str], *, show_progress: bool = False) -> np.ndarray:
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
            "Install it with:\n"
            "  pip install git+https://github.com/patcon/polis@package-commentgraph#subdirectory=delphi/umap_narrative/polismath_commentgraph"
        ) from exc

    return EmbeddingEngine().embed_batch(texts=texts, show_progress=show_progress)


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
            "Install it with:\n"
            "  pip install git+https://github.com/patcon/polis@package-commentgraph#subdirectory=delphi/umap_narrative/polismath_commentgraph"
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
            "Install it with:\n"
            "  pip install git+https://github.com/patcon/polis@package-commentgraph#subdirectory=delphi/umap_narrative/polismath_commentgraph"
        ) from exc

    return ClusteringEngine().create_clustering_layers(
        embeddings=embeddings, num_layers=num_layers
    )


def recipe_polis2_statements(adata: AnnData, *, show_progress: bool = False, inplace: bool = True) -> AnnData | None:
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
       and stores the result in ``.varm["content_embedding"]``.
    2. Projects the embeddings to 2-D with UMAP and stores the coordinates
       in ``.varm["content_umap"]``.
    3. Builds a hierarchy of clustering layers (finest → coarsest) and
       stores them in ``.varm["evoc_polis2"]`` (shape ``n_var × num_layers``)
       with the coarsest layer also surfaced as the categorical column
       ``.var["evoc_polis2_top"]``.

    Parameters
    ----------
    adata :
        AnnData object whose ``.var["content"]`` column contains the
        statement text strings.
    show_progress :
        Show embedding progress bar.  When ``False`` (the default),
        warnings and progress output from the model-loading libraries
        are also suppressed.
    inplace :
        If ``True`` (default), mutate *adata* and return ``None``.
        If ``False``, operate on a copy and return it.

    Returns
    -------
    Depending on *inplace*, returns ``None`` or the modified ``AnnData``.

    .varm['content_embedding']
        Dense text embeddings, shape ``(n_var, embed_dim)``.
    .varm['content_umap']
        2-D UMAP projection of the embeddings, shape ``(n_var, 2)``.
    .varm['evoc_polis2']
        Stacked layers of clustering labels, shape ``(n_var, num_layers)``.
        Column 0 is the finest/bottom; column -1 is the coarsest/top.  ``-1`` = noise.
    .var['evoc_polis2_top']
        Categorical column taken from the coarsest clustering layer
        (i.e. ``evoc_polis2[:, -1]``).

    Examples
    --------

    ```py
    adata = val.datasets.polis.chile_protests(translate_to="en")

    with val.viz.schematic_diagram(diff_from=adata):
        val.tools.recipe_polis2_statements(adata)

    val.viz.embedding(
        # Transpose .var and .obs axes for plotting
        adata.transpose(),
        basis="content_umap",
        color=["evoc_polis2_top", "moderation_state"],
    )
    ```
    """
    if not inplace:
        adata = adata.copy()

    texts = adata.var["content"].tolist()

    # Suppress noisy warnings / loggers from HF Hub, sentence-transformers
    # and umap during model loading, unless the caller opted into progress.
    ctx = _quiet() if not show_progress else contextmanager(lambda: (yield))()
    with ctx:
        adata.varm["content_embedding"] = _embed_statements(texts, show_progress=show_progress)
        adata.varm["content_umap"] = _project_umap(adata.varm["content_embedding"])
        cluster_layers = _create_cluster_layers(adata.varm["content_embedding"])

    adata.varm["evoc_polis2"] = np.array(cluster_layers).T
    adata.var["evoc_polis2_top"] = adata.varm["evoc_polis2"][:, -1]
    adata.var["evoc_polis2_top"] = (
        adata.var["evoc_polis2_top"]
        # -1 = noise/unassigned; convert to NA so scanpy renders as lightgray.
        .where(adata.var["evoc_polis2_top"] != -1)
        # Nullable int so NAs survive; category for discrete colormap.
        .astype("Int64")
        .astype("category")
    )

    if not inplace:
        return adata
