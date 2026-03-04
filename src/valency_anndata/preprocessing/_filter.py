from __future__ import annotations
from typing import Optional
import numpy as np
import anndata as ad


def filter_participants(
    data: ad.AnnData,
    min_statements: Optional[int] = None,
    *,
    inplace: bool = True,
) -> Optional[ad.AnnData]:
    """Filter participants (rows) based on vote counts.

    Counts the number of statements each participant has voted on (i.e.
    non-``NaN`` entries in the row). Unlike :func:`scanpy.pp.filter_cells`,
    this correctly treats ``0``, ``-1``, and ``+1`` all as real votes and
    only ``NaN`` as "no vote".

    Parameters
    ----------
    data
        Annotated data matrix (participants × statements).
    min_statements
        Minimum number of statements a participant must have voted on to be kept.
    inplace
        If ``True``, modifies ``data`` in place and returns ``None``.
        If ``False``, returns a filtered copy.

    Returns
    -------
    Filtered :class:`~anndata.AnnData` if ``inplace=False``, else ``None``.
    """
    vote_counts = np.sum(~np.isnan(data.X), axis=1)
    mask = np.ones(data.n_obs, dtype=bool)
    if min_statements is not None:
        mask &= vote_counts >= min_statements
    if inplace:
        data._inplace_subset_obs(mask)
        return None
    return data[mask].copy()


def filter_statements(
    data: ad.AnnData,
    min_participants: Optional[int] = None,
    *,
    inplace: bool = True,
) -> Optional[ad.AnnData]:
    """Filter statements (columns) based on vote counts.

    Counts the number of participants who voted on each statement (i.e.
    non-``NaN`` entries in the column). Unlike :func:`scanpy.pp.filter_genes`,
    this correctly treats ``0``, ``-1``, and ``+1`` all as real votes and
    only ``NaN`` as "no vote".

    Parameters
    ----------
    data
        Annotated data matrix (participants × statements).
    min_participants
        Minimum number of participants who must have voted on a statement
        for it to be kept.
    inplace
        If ``True``, modifies ``data`` in place and returns ``None``.
        If ``False``, returns a filtered copy.

    Returns
    -------
    Filtered :class:`~anndata.AnnData` if ``inplace=False``, else ``None``.
    """
    vote_counts = np.sum(~np.isnan(data.X), axis=0)
    mask = np.ones(data.n_vars, dtype=bool)
    if min_participants is not None:
        mask &= vote_counts >= min_participants
    if inplace:
        data._inplace_subset_var(mask)
        return None
    return data[:, mask].copy()
