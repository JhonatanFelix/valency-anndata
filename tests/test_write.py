"""Tests for valency_anndata.write (h5ad export with sanitization)."""

import anndata
import numpy as np
import pandas as pd
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


# ─────────────────────────────────────────────────────────────────────
# Include-filter tests
# ─────────────────────────────────────────────────────────────────────


def _make_rich_adata():
    """Create a small AnnData with obs, obsm, layers, and uns populated."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"kmeans_polis": ["0", "1", "0"], "batch": ["a", "b", "a"]},
            index=["p1", "p2", "p3"],
        ),
        var=pd.DataFrame(
            {"gene_name": ["g1", "g2"], "is_meta": [False, True]},
            index=["s1", "s2"],
        ),
    )
    adata.obsm["X_pca"] = np.ones((3, 2))
    adata.obsm["X_pacmap"] = np.ones((3, 2)) * 2
    adata.obsm["X_umap"] = np.ones((3, 2)) * 3
    adata.layers["X_masked"] = X.copy()
    adata.uns["statements"] = pd.DataFrame({"text": ["a", "b"]})
    adata.uns["recipe_config"] = {"k": 2}
    return adata


class TestWriteInclude:
    """val.write(include=...) filters the written file."""

    def test_include_exact_keys(self, tmp_path):
        """Only specified obsm/obs keys appear in output."""
        adata = _make_rich_adata()
        out = tmp_path / "exact.h5ad"
        val.write(out, adata, include=["obsm/X_pca", "obs/kmeans_polis"])

        reloaded = anndata.read_h5ad(out)
        assert list(reloaded.obsm.keys()) == ["X_pca"]
        assert "kmeans_polis" in reloaded.obs.columns
        assert "batch" not in reloaded.obs.columns

    def test_include_glob_pattern(self, tmp_path):
        """obsm/X_* matches all obsm keys starting with X_."""
        adata = _make_rich_adata()
        out = tmp_path / "glob.h5ad"
        val.write(out, adata, include=["obsm/X_*"])

        reloaded = anndata.read_h5ad(out)
        assert set(reloaded.obsm.keys()) == {"X_pca", "X_pacmap", "X_umap"}

    def test_include_multiple_namespaces(self, tmp_path):
        """Mix of obs/* and obsm/* patterns works."""
        adata = _make_rich_adata()
        out = tmp_path / "multi_ns.h5ad"
        val.write(
            out,
            adata,
            include=["obs/*", "obsm/X_pca", "layers/X_masked"],
        )

        reloaded = anndata.read_h5ad(out)
        assert "kmeans_polis" in reloaded.obs.columns
        assert "batch" in reloaded.obs.columns
        assert list(reloaded.obsm.keys()) == ["X_pca"]
        assert list(reloaded.layers.keys()) == ["X_masked"]

    def test_include_uns(self, tmp_path):
        """uns/* includes all uns keys."""
        adata = _make_rich_adata()
        out = tmp_path / "uns.h5ad"
        val.write(out, adata, include=["uns/*"])

        reloaded = anndata.read_h5ad(out)
        assert "statements" in reloaded.uns
        assert "recipe_config" in reloaded.uns
        # Other namespaces should be empty
        assert len(reloaded.obsm) == 0
        assert len(reloaded.layers) == 0

    def test_include_filters_out_unmatched(self, tmp_path):
        """Keys not in include are absent from written file."""
        adata = _make_rich_adata()
        out = tmp_path / "filtered.h5ad"
        val.write(out, adata, include=["obsm/X_pca"])

        reloaded = anndata.read_h5ad(out)
        assert "X_pacmap" not in reloaded.obsm
        assert "X_umap" not in reloaded.obsm
        assert len(reloaded.layers) == 0
        assert len(reloaded.uns) == 0
        assert len(reloaded.obs.columns) == 0

    def test_include_none_writes_everything(self, tmp_path):
        """Default behavior (include=None) writes all data."""
        adata = _make_rich_adata()
        out = tmp_path / "all.h5ad"
        val.write(out, adata, include=None)

        reloaded = anndata.read_h5ad(out)
        assert set(reloaded.obsm.keys()) == {"X_pca", "X_pacmap", "X_umap"}
        assert "kmeans_polis" in reloaded.obs.columns
        assert "batch" in reloaded.obs.columns
        assert "X_masked" in reloaded.layers
        assert "statements" in reloaded.uns
