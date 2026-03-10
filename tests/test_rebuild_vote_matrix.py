"""Tests for val.preprocessing.rebuild_vote_matrix()."""

import numpy as np
import pytest

from valency_anndata.datasets.polis import load
from valency_anndata.preprocessing import rebuild_vote_matrix


# The synthetic fixture has 16 raw vote rows from 5 voters across 4 statements.
# With trim_rule=0.5 (keep first 50% of rows by timestamp), voters 104 and 105
# are excluded, reducing shape from (5, 4) to (3, 4).
TRIM_RULE = 0.5
TRIMMED_SHAPE = (3, 4)
ORIGINAL_SHAPE = (5, 4)


class TestRebuildVoteMatrixTrimRule:
    def test_full_trim_preserves_shape(self, fixture_dir):
        adata = load(str(fixture_dir))
        rebuild_vote_matrix(adata, trim_rule=1.0)
        assert adata.shape == ORIGINAL_SHAPE

    def test_trim_reduces_shape(self, fixture_dir):
        adata = load(str(fixture_dir))
        assert adata.shape == ORIGINAL_SHAPE
        rebuild_vote_matrix(adata, trim_rule=TRIM_RULE)
        assert adata.shape == TRIMMED_SHAPE

    def test_trim_with_mismatched_layer_does_not_raise(self, fixture_dir):
        """Layers from the original (larger) matrix must not cause a ValueError
        when trimming produces a smaller AnnData."""
        adata = load(str(fixture_dir))
        # raw_sparse layer is added by load(); it has the original shape
        assert "raw_sparse" in adata.layers
        assert adata.layers["raw_sparse"].shape == ORIGINAL_SHAPE

        # This was raising ValueError before the fix
        rebuild_vote_matrix(adata, trim_rule=TRIM_RULE)
        assert adata.shape == TRIMMED_SHAPE

    def test_trim_drops_mismatched_layer(self, fixture_dir):
        """A layer that no longer fits the trimmed shape is silently dropped."""
        adata = load(str(fixture_dir))
        rebuild_vote_matrix(adata, trim_rule=TRIM_RULE)
        # raw_sparse was ORIGINAL_SHAPE; new shape is TRIMMED_SHAPE — should be gone
        assert "raw_sparse" not in adata.layers

    def test_trim_drops_mismatched_obsm(self, fixture_dir):
        """An obsm entry sized for the original n_obs is dropped after trim."""
        adata = load(str(fixture_dir))
        # Inject a fake embedding sized for the original 5 participants
        adata.obsm["X_pca"] = np.zeros((ORIGINAL_SHAPE[0], 2))
        rebuild_vote_matrix(adata, trim_rule=TRIM_RULE)
        assert "X_pca" not in adata.obsm

    def test_inplace_false_does_not_modify_original(self, fixture_dir):
        adata = load(str(fixture_dir))
        rebuild_vote_matrix(adata, trim_rule=TRIM_RULE, inplace=False)
        assert adata.shape == ORIGINAL_SHAPE

    def test_inplace_false_returns_trimmed_adata(self, fixture_dir):
        adata = load(str(fixture_dir))
        result = rebuild_vote_matrix(adata, trim_rule=TRIM_RULE, inplace=False)
        assert result is not None
        assert result is not adata
        assert result.shape == TRIMMED_SHAPE
