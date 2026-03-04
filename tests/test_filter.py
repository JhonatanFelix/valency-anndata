"""Tests for val.preprocessing.filter_participants() and filter_statements()."""

import numpy as np
import anndata as ad

from valency_anndata.preprocessing import filter_participants, filter_statements


def make_adata(X):
    return ad.AnnData(X=np.array(X, dtype=float))


class TestFilterParticipants:
    def test_removes_low_vote_participants(self):
        # Row 0: 2 votes, row 1: 1 vote, row 2: 2 votes
        adata = make_adata([
            [1.0, -1.0],
            [1.0, np.nan],
            [-1.0, 1.0],
        ])
        filter_participants(adata, min_statements=2)
        assert adata.n_obs == 2

    def test_keeps_all_participants_above_threshold(self):
        adata = make_adata([
            [1.0, -1.0],
            [-1.0, 1.0],
        ])
        filter_participants(adata, min_statements=2)
        assert adata.n_obs == 2

    def test_inplace_modifies_original(self):
        adata = make_adata([
            [1.0, -1.0],
            [np.nan, np.nan],
        ])
        filter_participants(adata, min_statements=1)
        assert adata.n_obs == 1

    def test_inplace_false_returns_copy(self):
        adata = make_adata([
            [1.0, -1.0],
            [np.nan, np.nan],
        ])
        result = filter_participants(adata, min_statements=1, inplace=False)
        assert result.n_obs == 1
        assert adata.n_obs == 2  # original unchanged

    def test_shape_trimmed_correctly(self):
        # 4 participants, 3 statements; keep only those with >= 2 votes
        adata = make_adata([
            [1.0, -1.0, 1.0],   # 3 votes — keep
            [np.nan, np.nan, np.nan],  # 0 votes — drop
            [1.0, np.nan, -1.0],  # 2 votes — keep
            [np.nan, 1.0, np.nan],  # 1 vote — drop
        ])
        filter_participants(adata, min_statements=2)
        assert adata.shape == (2, 3)


class TestFilterStatements:
    def test_removes_low_vote_statements(self):
        # Col 0: 2 votes, col 1: 1 vote
        adata = make_adata([
            [1.0, -1.0],
            [-1.0, np.nan],
        ])
        filter_statements(adata, min_participants=2)
        assert adata.n_vars == 1

    def test_keeps_all_statements_above_threshold(self):
        adata = make_adata([
            [1.0, -1.0],
            [-1.0, 1.0],
        ])
        filter_statements(adata, min_participants=2)
        assert adata.n_vars == 2

    def test_inplace_modifies_original(self):
        adata = make_adata([
            [1.0, np.nan],
            [-1.0, np.nan],
        ])
        filter_statements(adata, min_participants=1)
        assert adata.n_vars == 1

    def test_inplace_false_returns_copy(self):
        adata = make_adata([
            [1.0, np.nan],
            [-1.0, np.nan],
        ])
        result = filter_statements(adata, min_participants=1, inplace=False)
        assert result.n_vars == 1
        assert adata.n_vars == 2  # original unchanged

    def test_shape_trimmed_correctly(self):
        # 3 participants, 4 statements; keep only statements with >= 2 votes
        # col 0: 2 votes, col 1: 0 votes, col 2: 3 votes, col 3: 2 votes → drop col 1
        adata = make_adata([
            [1.0, np.nan, -1.0, 1.0],
            [-1.0, np.nan, 1.0, np.nan],
            [np.nan, np.nan, -1.0, 1.0],
        ])
        filter_statements(adata, min_participants=2)
        assert adata.shape == (3, 3)
