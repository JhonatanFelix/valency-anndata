"""Export AnnData objects to disk with automatic sanitization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import scanpy as sc

if TYPE_CHECKING:
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

    # --- force cluster labels to strings ---
    for col in [c for c in adata.obs.columns if c.startswith("kmeans_")]:
        adata.obs[col] = (
            adata.obs[col]
            .astype(object)
            .infer_objects(copy=False)
            .fillna(-2)
            .astype(str)
        )

    return adata


def write(
    filename: Path | str,
    adata: AnnData,
    *,
    ext: Literal["h5", "csv", "txt", "npz"] | None = None,
    compression: Literal["gzip", "lzf"] | None = "gzip",
    compression_opts: int | None = None,
) -> None:
    """Write an :class:`~anndata.AnnData` object to file with automatic sanitization.

    Wraps :func:`scanpy.write` but first copies and sanitizes *adata* so that
    problematic fields (mixed-type ``uns["statements"]`` columns, categorical
    ``kmeans_*`` labels with ``NA``) do not cause serialization errors.

    Parameters
    ----------
    filename
        Output path.  If the filename has no file extension it is interpreted
        the same way as :func:`scanpy.write`.
    adata
        Annotated data matrix.  **Not** mutated — a sanitized copy is written.
    ext
        File extension from which to infer file format.
    compression
        See `h5py dataset docs <https://docs.h5py.org/en/latest/high/dataset.html>`_.
    compression_opts
        See `h5py dataset docs <https://docs.h5py.org/en/latest/high/dataset.html>`_.
    """
    sanitized = _sanitize_for_export(adata)
    sc.write(
        filename,
        sanitized,
        ext=ext,
        compression=compression,
        compression_opts=compression_opts,
    )
