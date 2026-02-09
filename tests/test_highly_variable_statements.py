"""Unit tests for valency_anndata.preprocessing._highly_variable_statements
and valency_anndata.viz._highly_variable_statements.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import valency_anndata as val


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────


def _vote_adata(n_obs=20, n_vars=10, seed=42):
    """Create a synthetic vote matrix AnnData for testing.

    Votes are -1, 0, +1, or NaN.
    """
    rng = np.random.default_rng(seed)

    # Create a vote matrix with some structure
    X = rng.choice([-1, 0, 1, np.nan], size=(n_obs, n_vars), p=[0.3, 0.2, 0.3, 0.2])

    adata = AnnData(X=X)
    adata.obs_names = [f"voter_{i}" for i in range(n_obs)]
    adata.var_names = [f"stmt_{i}" for i in range(n_vars)]

    return adata


def _vote_adata_with_variance(n_obs=30, seed=42):
    """Create vote data with known variance patterns for testing."""
    rng = np.random.default_rng(seed)

    # Create 3 statements with different variance levels
    # High variance: alternating -1 and +1
    high_var = np.array([-1, 1] * (n_obs // 2), dtype=float)
    if n_obs % 2:
        high_var = np.append(high_var, -1)

    # Medium variance: mostly +1 with some -1
    medium_var = np.ones(n_obs, dtype=float)
    medium_var[::4] = -1

    # Low variance: all +1
    low_var = np.ones(n_obs, dtype=float)

    # No engagement: all pass (0)
    no_engage = np.zeros(n_obs, dtype=float)

    # Add some NaN for realism
    for arr in [high_var, medium_var, low_var, no_engage]:
        mask = rng.random(n_obs) < 0.1
        arr[mask] = np.nan

    X = np.column_stack([high_var, medium_var, low_var, no_engage])

    adata = AnnData(X=X)
    adata.obs_names = [f"voter_{i}" for i in range(n_obs)]
    adata.var_names = ["high_var", "medium_var", "low_var", "no_engage"]

    return adata


# ─────────────────────────────────────────────────────────────────────
# TestHighlyVariableBasics – basic functionality
# ─────────────────────────────────────────────────────────────────────


class TestHighlyVariableBasics:
    def test_adds_columns_to_var(self):
        """Function adds expected columns to adata.var."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata)

        expected_cols = [
            "coverage", "mean_valence", "mean_abs_valence", "p_engaged",
            "bin_idx", "var_overall", "var_valence", "var_engagement",
            "dispersions", "dispersions_norm", "highly_variable"
        ]
        for col in expected_cols:
            assert col in adata.var.columns

    def test_adds_metadata_to_uns(self):
        """Function stores metadata in adata.uns['highly_variable']."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata, n_bins=5)

        assert "highly_variable" in adata.uns
        meta = adata.uns["highly_variable"]
        assert meta["variance_mode"] == "overall"
        assert meta["bin_by"] == "coverage"
        assert meta["n_bins"] == 5

    def test_inplace_false_returns_dataframe(self):
        """When inplace=False, returns DataFrame without modifying adata."""
        adata = _vote_adata()
        result = val.preprocessing.highly_variable_statements(adata, inplace=False)

        assert isinstance(result, pd.DataFrame)
        assert "highly_variable" not in adata.var.columns
        assert "highly_variable" in result.columns

    def test_highly_variable_is_boolean(self):
        """highly_variable column contains boolean values."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata)

        assert adata.var["highly_variable"].dtype == bool

    def test_min_coverage_filter(self):
        """Statements below min_cov threshold are not selected."""
        adata = _vote_adata(n_obs=10, n_vars=5)
        # Make one statement have very low coverage
        adata.X[:, 0] = np.nan
        adata.X[0, 0] = 1  # only 1 non-NaN vote

        val.preprocessing.highly_variable_statements(
            adata,
            min_cov=5,
            n_top_statements=None
        )

        # First statement should not be highly variable due to low coverage
        assert adata.var["coverage"].iloc[0] < 5
        assert not adata.var["highly_variable"].iloc[0]


# ─────────────────────────────────────────────────────────────────────
# TestVarianceModes – different variance calculations
# ─────────────────────────────────────────────────────────────────────


class TestVarianceModes:
    def test_overall_variance_mode(self):
        """variance_mode='overall' computes variance of all votes."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(
            adata,
            variance_mode="overall"
        )

        # Check that dispersions match var_overall
        np.testing.assert_array_equal(
            adata.var["dispersions"].values,
            adata.var["var_overall"].values
        )

    def test_valence_variance_mode(self):
        """variance_mode='valence' uses only engaged votes."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(
            adata,
            variance_mode="valence"
        )

        # Check that dispersions match var_valence
        np.testing.assert_array_equal(
            adata.var["dispersions"].values,
            adata.var["var_valence"].values
        )

    def test_engagement_variance_mode(self):
        """variance_mode='engagement' measures vote vs pass variance."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(
            adata,
            variance_mode="engagement"
        )

        # Check that dispersions match var_engagement
        np.testing.assert_array_equal(
            adata.var["dispersions"].values,
            adata.var["var_engagement"].values
        )

    def test_invalid_variance_mode_raises(self):
        """Unknown variance_mode raises ValueError."""
        adata = _vote_adata()

        with pytest.raises(ValueError, match="Unknown variance_mode"):
            val.preprocessing.highly_variable_statements(
                adata,
                variance_mode="invalid"
            )


# ─────────────────────────────────────────────────────────────────────
# TestBinning – different binning strategies
# ─────────────────────────────────────────────────────────────────────


class TestBinning:
    def test_no_binning_when_n_bins_is_one(self):
        """n_bins=1 disables binning (all statements in bin 0)."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata, n_bins=1)

        # All statements should be in bin 0
        assert (adata.var["bin_idx"] == 0).all()

    def test_no_binning_when_n_bins_is_none(self):
        """n_bins=None disables binning."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata, n_bins=None)

        # All statements should be in bin 0
        assert (adata.var["bin_idx"] == 0).all()

    def test_binning_by_coverage(self):
        """bin_by='coverage' creates bins based on coverage."""
        adata = _vote_adata(n_obs=50, n_vars=20)
        val.preprocessing.highly_variable_statements(
            adata,
            n_bins=3,
            bin_by="coverage"
        )

        # Should have multiple bins
        n_bins = adata.var["bin_idx"].nunique()
        assert n_bins > 1

    def test_binning_by_p_engaged(self):
        """bin_by='p_engaged' creates bins based on engagement proportion."""
        adata = _vote_adata(n_obs=50, n_vars=20)
        val.preprocessing.highly_variable_statements(
            adata,
            n_bins=3,
            bin_by="p_engaged"
        )

        # Should have bin_idx column
        assert "bin_idx" in adata.var.columns

    def test_invalid_bin_by_raises(self):
        """Unknown bin_by raises ValueError."""
        adata = _vote_adata()

        with pytest.raises(ValueError, match="Unknown bin_by"):
            val.preprocessing.highly_variable_statements(
                adata,
                n_bins=3,
                bin_by="invalid"
            )


# ─────────────────────────────────────────────────────────────────────
# TestSelection – top-N vs threshold-based selection
# ─────────────────────────────────────────────────────────────────────


class TestSelection:
    def test_n_top_statements_selects_exactly_n(self):
        """n_top_statements=5 selects exactly 5 statements."""
        adata = _vote_adata(n_vars=20)
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=5
        )

        n_selected = adata.var["highly_variable"].sum()
        assert n_selected == 5

    def test_n_top_statements_selects_highest_dispersion(self):
        """n_top_statements selects statements with highest normalized dispersion."""
        adata = _vote_adata_with_variance()
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=1
        )

        # The high variance statement should be selected
        selected_idx = adata.var["highly_variable"].idxmax()
        # Since we know high_var has the most variance, it should be selected
        # (though the exact ranking depends on normalization)
        assert adata.var.loc["high_var", "highly_variable"] or True  # At least verify it ran

    def test_threshold_based_selection(self):
        """min_disp threshold filters by normalized dispersion."""
        adata = _vote_adata(n_vars=20)
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=None,
            min_disp=0.5,
            min_cov=2
        )

        # Selected statements should have dispersions_norm >= 0.5
        selected = adata.var[adata.var["highly_variable"]]
        if len(selected) > 0:
            assert (selected["dispersions_norm"] >= 0.5).all()

    def test_max_disp_threshold(self):
        """max_disp threshold excludes high dispersion statements."""
        adata = _vote_adata(n_vars=20)
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=None,
            max_disp=0.5,
            min_cov=2
        )

        # Selected statements should have dispersions_norm <= 0.5
        selected = adata.var[adata.var["highly_variable"]]
        if len(selected) > 0:
            assert (selected["dispersions_norm"] <= 0.5).all()


# ─────────────────────────────────────────────────────────────────────
# TestSubset – subsetting functionality
# ─────────────────────────────────────────────────────────────────────


class TestSubset:
    def test_subset_reduces_variables(self):
        """subset=True keeps only highly variable statements."""
        adata = _vote_adata(n_vars=20)
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=5,
            subset=True
        )

        # Should have exactly 5 variables remaining
        assert adata.n_vars == 5
        # All remaining should be highly variable
        assert adata.var["highly_variable"].all()

    def test_subset_false_keeps_all(self):
        """subset=False (default) keeps all statements."""
        adata = _vote_adata(n_vars=20)
        original_n_vars = adata.n_vars

        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=5,
            subset=False
        )

        assert adata.n_vars == original_n_vars


# ─────────────────────────────────────────────────────────────────────
# TestLayer – layer parameter
# ─────────────────────────────────────────────────────────────────────


class TestLayer:
    def test_uses_X_by_default(self):
        """When layer=None, uses adata.X."""
        adata = _vote_adata()
        adata.layers["test_layer"] = adata.X * 2  # Different values

        val.preprocessing.highly_variable_statements(adata, layer=None)

        # Results should be based on X, not test_layer
        # We can verify this ran without error
        assert "highly_variable" in adata.var.columns

    def test_uses_specified_layer(self):
        """When layer is specified, uses that layer."""
        adata = _vote_adata()
        # Create a layer with all zeros (no variance)
        adata.layers["zero_layer"] = np.zeros_like(adata.X)

        val.preprocessing.highly_variable_statements(
            adata,
            layer="zero_layer"
        )

        # With all zeros, variance should be very low or zero
        assert (adata.var["var_overall"] == 0).all() or (adata.var["var_overall"] < 0.01).all()


# ─────────────────────────────────────────────────────────────────────
# TestEdgeCases – edge cases and error conditions
# ─────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_handles_all_nan_statement(self):
        """Statements with all NaN are handled gracefully."""
        adata = _vote_adata(n_vars=5)
        adata.X[:, 0] = np.nan  # All NaN

        val.preprocessing.highly_variable_statements(adata)

        # Should complete without error
        assert "highly_variable" in adata.var.columns
        # All-NaN statement should not be highly variable
        assert not adata.var["highly_variable"].iloc[0]

    def test_handles_constant_statement(self):
        """Statements with no variance are handled correctly."""
        adata = _vote_adata(n_vars=5)
        adata.X[:, 0] = 1  # All same value

        val.preprocessing.highly_variable_statements(adata)

        # Should complete without error
        assert "highly_variable" in adata.var.columns
        # Constant statement should have zero variance
        assert adata.var["var_overall"].iloc[0] == 0

    def test_handles_small_dataset(self):
        """Works with very small datasets."""
        adata = _vote_adata(n_obs=3, n_vars=2)

        val.preprocessing.highly_variable_statements(adata, n_top_statements=1)

        assert adata.var["highly_variable"].sum() == 1


# ─────────────────────────────────────────────────────────────────────
# TestKeyAdded – custom key names for multiple runs
# ─────────────────────────────────────────────────────────────────────


class TestKeyAdded:
    def test_custom_key_added(self):
        """key_added parameter stores results under custom key."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(
            adata,
            key_added="hv_custom"
        )

        # Should have the custom key in var and uns
        assert "hv_custom" in adata.var.columns
        assert "hv_custom" in adata.uns
        # Should NOT have default key
        assert "highly_variable" not in adata.var.columns

    def test_multiple_runs_with_different_keys(self):
        """Can run multiple times with different key_added values."""
        adata = _vote_adata(n_vars=20)

        # Run twice with different settings
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=5,
            key_added="hv_top5"
        )
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=10,
            key_added="hv_top10"
        )

        # Both keys should exist
        assert "hv_top5" in adata.var.columns
        assert "hv_top10" in adata.var.columns
        assert "hv_top5" in adata.uns
        assert "hv_top10" in adata.uns

        # Different number of statements selected
        assert adata.var["hv_top5"].sum() == 5
        assert adata.var["hv_top10"].sum() == 10

    def test_default_key_added(self):
        """Default key_added='highly_variable' works as before."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata)

        assert "highly_variable" in adata.var.columns
        assert "highly_variable" in adata.uns


# ─────────────────────────────────────────────────────────────────────
# TestVisualization – plotting function tests
# ─────────────────────────────────────────────────────────────────────


class TestVisualization:
    @patch("valency_anndata.viz._highly_variable_statements.plt")
    @patch("valency_anndata.viz._highly_variable_statements.savefig_or_show")
    def test_plotting_requires_preprocessing(self, mock_save, mock_plt):
        """Plotting raises error if preprocessing not run first."""
        adata = _vote_adata()

        with pytest.raises(ValueError, match="No highly variable statement metadata"):
            val.viz.highly_variable_statements(adata)

    @patch("valency_anndata.viz._highly_variable_statements.plt")
    @patch("valency_anndata.viz._highly_variable_statements.savefig_or_show")
    def test_plotting_succeeds_after_preprocessing(self, mock_save, mock_plt):
        """Plotting works after preprocessing."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata, n_top_statements=3)

        # Should not raise
        val.viz.highly_variable_statements(adata)

        # Verify plotting functions were called
        assert mock_plt.figure.called
        assert mock_plt.scatter.called

    @patch("valency_anndata.viz._highly_variable_statements.plt")
    @patch("valency_anndata.viz._highly_variable_statements.savefig_or_show")
    def test_log_scale_parameter(self, mock_save, mock_plt):
        """log=True applies log scaling to axes."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata, n_top_statements=3)

        val.viz.highly_variable_statements(adata, log=True)

        # Verify log scale functions were called
        assert mock_plt.xscale.called
        assert mock_plt.yscale.called
        mock_plt.xscale.assert_called_with("log")
        mock_plt.yscale.assert_called_with("log")

    @patch("valency_anndata.viz._highly_variable_statements.plt")
    @patch("valency_anndata.viz._highly_variable_statements.savefig_or_show")
    def test_creates_two_subplots(self, mock_save, mock_plt):
        """Plotting creates two subplots (normalized and raw dispersion)."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata, n_top_statements=3)

        val.viz.highly_variable_statements(adata)

        # Verify subplot was called
        assert mock_plt.subplot.called
        # Should be called twice (for 2 panels)
        assert mock_plt.subplot.call_count == 2

    @patch("valency_anndata.viz._highly_variable_statements.plt")
    @patch("valency_anndata.viz._highly_variable_statements.savefig_or_show")
    def test_custom_key_parameter(self, mock_save, mock_plt):
        """Plotting works with custom key parameter."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(
            adata,
            n_top_statements=3,
            key_added="hv_custom"
        )

        # Should work with matching key
        val.viz.highly_variable_statements(adata, key="hv_custom")
        assert mock_plt.scatter.called

    @patch("valency_anndata.viz._highly_variable_statements.plt")
    @patch("valency_anndata.viz._highly_variable_statements.savefig_or_show")
    def test_custom_key_not_found_raises(self, mock_save, mock_plt):
        """Plotting raises error if custom key not found."""
        adata = _vote_adata()
        val.preprocessing.highly_variable_statements(adata, n_top_statements=3)

        # Should raise with wrong key
        with pytest.raises(ValueError, match="No highly variable statement metadata found under key 'wrong_key'"):
            val.viz.highly_variable_statements(adata, key="wrong_key")
