"""Export AnnData objects to disk with automatic sanitization."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING, Literal

import pandas as pd
import scanpy as sc

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from anndata import AnnData


def _coerce_object_columns(df: pd.DataFrame) -> None:
    """Coerce object-dtype columns containing None/NA to strings in-place."""
    for col in df.columns:
        if df[col].dtype == object and df[col].isna().any():
            df[col] = df[col].map(lambda x: "" if x is None or pd.isna(x) else x)


def _sanitize_for_export(adata: AnnData) -> AnnData:
    """Make an AnnData copy safe for h5ad serialization.

    Ported from ``patcon/streamlit-valency-anndata`` and extended to cover
    additional edge-cases discovered in local-fixture tests.
    """
    adata = adata.copy()

    # --- coerce object columns with None/NA in uns DataFrames ---
    for key in list(adata.uns):
        if isinstance(adata.uns[key], pd.DataFrame):
            _coerce_object_columns(adata.uns[key])

    # --- coerce object columns with None/NA in obs and var ---
    _coerce_object_columns(adata.obs)
    _coerce_object_columns(adata.var)

    return adata


def _filter_adata(adata: AnnData, include: Sequence[str]) -> AnnData:
    """Keep only the keys matched by *include* patterns.

    Each element of *include* has the form ``"namespace/pattern"`` where
    *namespace* is one of ``obs``, ``var``, ``obsm``, ``varm``, ``layers``,
    ``uns``, ``obsp``, ``varp`` and *pattern* is an [`fnmatch`][fnmatch] glob
    matched against the keys within that namespace.

    ``X`` and ``raw`` are always retained.  Index columns of ``obs`` / ``var``
    are never stripped.

    Parameters
    ----------
    adata
        A **copy** of the annotated data matrix (mutated in-place).
    include
        Glob-style paths, e.g. ``["obsm/X_*", "obs/kmeans_*"]``.

    Returns
    -------
    The same `adata` object, filtered in-place for convenience.
    """
    # Parse include paths into {namespace: [patterns]}
    ns_patterns: dict[str, list[str]] = {}
    for path in include:
        ns, _, pattern = path.partition("/")
        ns_patterns.setdefault(ns, []).append(pattern)

    # Namespaces that hold dict-like mappings
    dict_namespaces = ("obsm", "varm", "layers", "uns", "obsp", "varp")

    # Filter DataFrame-based namespaces (obs, var): drop unmatched columns
    for ns in ("obs", "var"):
        patterns = ns_patterns.get(ns)
        if patterns is None:
            # Namespace not mentioned at all — drop all non-index columns
            df = getattr(adata, ns)
            drop_cols = list(df.columns)
        else:
            df = getattr(adata, ns)
            drop_cols = [
                col
                for col in df.columns
                if not any(fnmatch(col, p) for p in patterns)
            ]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    # Filter dict-like namespaces
    for ns in dict_namespaces:
        mapping = getattr(adata, ns)
        patterns = ns_patterns.get(ns)
        if patterns is None:
            # Namespace not mentioned — remove all keys
            for key in list(mapping.keys()):
                del mapping[key]
        else:
            for key in list(mapping.keys()):
                if not any(fnmatch(key, p) for p in patterns):
                    del mapping[key]

    return adata


def write(
    filename: Path | str,
    adata: AnnData,
    *,
    include: Sequence[str] | None = None,
    ext: Literal["h5", "csv", "txt", "npz"] | None = None,
    compression: Literal["gzip", "lzf"] | None = "gzip",
    compression_opts: int | None = None,
) -> None:
    """Write an [AnnData][anndata.AnnData] object to file with automatic sanitization.

    Wraps [scanpy.write][] but first copies and sanitizes `adata` so that
    problematic fields (mixed-type ``uns["statements"]`` columns) do not
    cause serialization errors.

    Parameters
    ----------
    filename
        Output path.  If the filename has no file extension it is interpreted
        the same way as [scanpy.write][].
    adata
        Annotated data matrix.  **Not** mutated — a sanitized copy is written.
    include
        When not ``None``, only the listed ``"namespace/key"`` paths are kept
        in the written file.  Glob patterns are supported
        (e.g. ``"obsm/X_*"``, ``"obs/kmeans_*"``).  Valid namespaces are
        ``obs``, ``var``, ``obsm``, ``varm``, ``layers``, ``uns``, ``obsp``,
        and ``varp``.  ``X`` and ``raw`` are always retained.
    ext
        File extension from which to infer file format.
    compression
        See `h5py dataset docs <https://docs.h5py.org/en/latest/high/dataset.html>`_.
    compression_opts
        See `h5py dataset docs <https://docs.h5py.org/en/latest/high/dataset.html>`_.

    Notes
    -----
    **Cluster labels and missing values.**
    Clustering columns (``kmeans_*``) are stored as
    `categorical arrays <https://anndata.readthedocs.io/en/latest/fileformat-prose.html>`_
    in the h5ad file. The on-disk encoding uses integer *codes* that index
    into a *categories* array, with ``-1`` reserved for missing entries.
    This means two distinct "absent" semantics survive the round-trip:

    * **Label ``-1``** (e.g. HDBSCAN noise points) is a real category in
      the categories array. It is a valid cluster assignment meaning
      "this participant was clustered but not assigned to any group."
    * **``NaN`` / ``pd.NA``** means the participant was never part of the
      clustering subset (e.g. excluded by ``mask_obs``). On disk this is
      represented by code ``-1``, which points to no category.

    After reading the file back with :func:`anndata.read_h5ad`, you can
    distinguish the two with :func:`pandas.isna`::

        labels = adata.obs["kmeans_polis"]
        noise  = labels == -1     # clustered, but no group
        unseen = labels.isna()    # not in the clustering subset

    Examples
    --------
    Basic — write everything:

    ```py
    val.write("conversation.h5ad", adata)
    ```

    Advanced — selectively include keys with glob patterns:

    ```py
    val.write(
        "export.h5ad",
        adata,
        include=["obsm/X_pca", "obsm/X_pacmap", "obs/kmeans_*", "uns/*"],
    )
    ```
    """
    sanitized = _sanitize_for_export(adata)
    if include is not None:
        _filter_adata(sanitized, include)
    sc.write(
        filename,
        sanitized,
        ext=ext,
        compression=compression,
        compression_opts=compression_opts,
    )
