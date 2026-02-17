"""Tests for valency_anndata.tools._polis.recipe_polis."""

import numpy as np
from anndata import AnnData

import valency_anndata as val


def _simple_vote_adata(n_obs=30, n_vars=15, seed=42):
    """Create a simple vote matrix for recipe_polis testing."""
    rng = np.random.default_rng(seed)

    # Create vote matrix
    X = rng.choice([-1, 0, 1, np.nan], size=(n_obs, n_vars), p=[0.3, 0.2, 0.3, 0.2])

    adata = AnnData(X=X)
    adata.obs_names = [f"voter_{i}" for i in range(n_obs)]
    adata.var_names = [f"stmt_{i}" for i in range(n_vars)]

    # Add required metadata for recipe_polis
    adata.var["is_meta"] = False
    adata.var["moderation_state"] = 1

    return adata


class TestRecipePolisMaskVar:
    """Tests for mask_var parameter in recipe_polis."""

    def test_recipe_polis_with_mask_var_none(self):
        """recipe_polis works with mask_var=None (default)."""
        adata = _simple_vote_adata()

        # Should not raise
        val.tools.recipe_polis(adata, mask_var=None)

        # Should have created expected outputs
        assert "X_pca_polis" in adata.obsm
        assert "kmeans_polis" in adata.obs

    def test_recipe_polis_ignores_highly_variable_when_mask_var_none(self):
        """recipe_polis with mask_var=None uses all statements even when highly_variable exists."""
        adata = _simple_vote_adata()

        # Add a highly_variable column that selects only a subset
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=5,
            key_added="highly_variable"
        )

        # Run with default mask_var=None — should use ALL statements, not just HVGs
        val.tools.recipe_polis(adata, mask_var=None)

        # PCA should have used all statements (n_vars), not just the 5 highly variable ones
        n_pcs = adata.obsm["X_pca_polis"].shape[1]
        n_vars = adata.shape[1]
        n_hvg = adata.var["highly_variable"].sum()
        # If PCA auto-detected highly_variable, max PCs would be limited by n_hvg.
        # With all statements, max PCs can be up to min(n_obs, n_vars).
        assert n_pcs <= min(adata.shape[0], n_vars)
        assert n_pcs > n_hvg, (
            f"PCA produced only {n_pcs} components (same as {n_hvg} HVGs) — "
            "highly_variable may have been auto-detected despite mask_var=None"
        )

    def test_recipe_polis_with_highly_variable_mask(self):
        """recipe_polis works with mask_var='highly_variable'."""
        adata = _simple_vote_adata()

        # First identify highly variable statements
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=8,
            key_added="highly_variable"
        )

        # Should not raise with mask_var
        val.tools.recipe_polis(adata, mask_var="highly_variable")

        # Should have created expected outputs
        assert "X_pca_polis" in adata.obsm
        assert "kmeans_polis" in adata.obs

    def test_recipe_polis_with_custom_mask_var(self):
        """recipe_polis works with custom mask_var names."""
        adata = _simple_vote_adata()

        # Create custom mask with different name
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=10,
            key_added="hv_top10"
        )

        # Should work with custom mask name
        val.tools.recipe_polis(adata, mask_var="hv_top10")

        # Should have created expected outputs
        assert "X_pca_polis" in adata.obsm
        assert "kmeans_polis" in adata.obs
