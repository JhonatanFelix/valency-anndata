"""Tests for valency_anndata.write (h5ad export with sanitization)."""

import anndata
import numpy as np
import pytest

import valency_anndata as val
from valency_anndata.datasets.polis import load


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def polis_adata(real_fixture_dir):
    """Load a real Polis dataset from local fixtures."""
    return load(str(real_fixture_dir))


def _prepare_for_recipe(adata):
    """Fill NA in is_meta/moderation_state so recipe_polis can run.

    The real fixture's CSV export lacks these columns, so load() sets them
    to NA.  recipe_polis requires concrete values.
    """
    import pandas as pd

    # https://stackoverflow.com/a/78066237
    with pd.option_context("future.no_silent_downcasting", True):
        adata.var["is_meta"] = adata.var["is_meta"].fillna(False).infer_objects(copy=False)


# ─────────────────────────────────────────────────────────────────────
# Write after each pipeline function
# ─────────────────────────────────────────────────────────────────────


class TestWriteAfterPipelineSteps:
    """val.write() succeeds after each major pipeline function."""

    def test_write_raw_loaded_data(self, polis_adata, tmp_path):
        """Write immediately after loading raw data."""
        out = tmp_path / "raw.h5ad"
        val.write(out, polis_adata)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_after_recipe_polis(self, polis_adata, tmp_path):
        """Write after running the full Polis pipeline."""
        _prepare_for_recipe(polis_adata)
        val.tools.recipe_polis(polis_adata)
        out = tmp_path / "recipe.h5ad"
        val.write(out, polis_adata)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_after_calculate_qc_metrics(self, polis_adata, tmp_path):
        """Write after computing QC metrics."""
        val.preprocessing.calculate_qc_metrics(polis_adata, inplace=True)
        out = tmp_path / "qc.h5ad"
        val.write(out, polis_adata)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_after_highly_variable_statements(self, polis_adata, tmp_path):
        """Write after identifying highly variable statements."""
        val.preprocessing.highly_variable_statements(polis_adata, n_top_statements=5)
        out = tmp_path / "hvs.h5ad"
        val.write(out, polis_adata)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_after_pacmap(self, polis_adata, tmp_path):
        """Write after running PaCMAP (requires imputed layer from recipe_polis)."""
        _prepare_for_recipe(polis_adata)
        val.tools.recipe_polis(polis_adata)
        val.tools.pacmap(polis_adata, layer="X_masked_imputed_mean")
        out = tmp_path / "pacmap.h5ad"
        val.write(out, polis_adata)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_after_localmap(self, polis_adata, tmp_path):
        """Write after running LocalMAP (requires imputed layer from recipe_polis)."""
        _prepare_for_recipe(polis_adata)
        val.tools.recipe_polis(polis_adata)
        val.tools.localmap(polis_adata, layer="X_masked_imputed_mean")
        out = tmp_path / "localmap.h5ad"
        val.write(out, polis_adata)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_after_rebuild_vote_matrix(self, polis_adata, tmp_path):
        """Write after rebuilding the vote matrix."""
        val.preprocessing.rebuild_vote_matrix(polis_adata)
        out = tmp_path / "rebuilt.h5ad"
        val.write(out, polis_adata)
        assert out.exists()
        assert out.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────
# Behaviour tests
# ─────────────────────────────────────────────────────────────────────


class TestWriteBehaviour:
    """val.write() does not mutate the original and produces readable files."""

    def test_no_mutation(self, polis_adata, tmp_path):
        """Original adata is not mutated by val.write()."""
        _prepare_for_recipe(polis_adata)
        val.tools.recipe_polis(polis_adata)

        # Snapshot obs dtypes and uns keys before write
        obs_dtypes_before = polis_adata.obs.dtypes.copy()
        uns_keys_before = set(polis_adata.uns.keys())
        obs_values_before = polis_adata.obs.copy()

        out = tmp_path / "nomut.h5ad"
        val.write(out, polis_adata)

        # obs dtypes unchanged
        assert polis_adata.obs.dtypes.equals(obs_dtypes_before)
        # uns keys unchanged
        assert set(polis_adata.uns.keys()) == uns_keys_before
        # obs values unchanged (including NaN equality)
        assert polis_adata.obs.equals(obs_values_before)

    def test_round_trip(self, polis_adata, tmp_path):
        """Written file can be read back with anndata.read_h5ad()."""
        _prepare_for_recipe(polis_adata)
        val.tools.recipe_polis(polis_adata)
        out = tmp_path / "roundtrip.h5ad"
        val.write(out, polis_adata)

        reloaded = anndata.read_h5ad(out)
        assert reloaded.n_obs == polis_adata.n_obs
        assert reloaded.n_vars == polis_adata.n_vars
        # Vote matrix values should match (NaN-aware)
        np.testing.assert_array_equal(
            np.isnan(reloaded.X), np.isnan(polis_adata.X)
        )
