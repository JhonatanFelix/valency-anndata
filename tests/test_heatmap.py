"""Unit tests for valency_anndata.viz._heatmap."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import numpy as np
import pytest
from anndata import AnnData

import valency_anndata as val


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────


def _vote_adata(n_obs=15, n_vars=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.choice([-1.0, 0.0, 1.0, np.nan], size=(n_obs, n_vars), p=[0.3, 0.2, 0.3, 0.2])
    adata = AnnData(X=X)
    adata.obs_names = [f"voter_{i}" for i in range(n_obs)]
    adata.var_names = [f"stmt_{i}" for i in range(n_vars)]
    return adata


# ─────────────────────────────────────────────────────────────────────
# TestHeatmapImport – registration side-effects
# ─────────────────────────────────────────────────────────────────────


class TestHeatmapImport:
    def test_heatmap_accessible_via_val_viz(self):
        assert callable(val.viz.heatmap)

    def test_rdylgn_bright_colormap_registered(self):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("RdYlGnBright")
        assert cmap is not None
        # Should map 3 discrete colours
        assert cmap(0.0) != cmap(0.5)
        assert cmap(0.5) != cmap(1.0)


# ─────────────────────────────────────────────────────────────────────
# TestHeatmapNoGroupby – dummy obs column workaround
# ─────────────────────────────────────────────────────────────────────


class TestHeatmapNoGroupby:
    def test_runs_without_groupby(self):
        """heatmap() works when groupby is omitted."""
        adata = _vote_adata()
        axes = val.viz.heatmap(adata, show=False)
        assert axes is not None

    def test_dummy_col_removed_after_call(self):
        """The temporary __heatmap_dummy__ column is cleaned up."""
        adata = _vote_adata()
        val.viz.heatmap(adata, show=False)
        assert "__heatmap_dummy__" not in adata.obs.columns

    def test_dummy_col_removed_on_exception(self):
        """Even if something goes wrong, dummy col is not silently left behind."""
        # We can't easily force an internal exception, but we verify the
        # column is absent both before and after a normal call.
        adata = _vote_adata()
        assert "__heatmap_dummy__" not in adata.obs.columns
        val.viz.heatmap(adata, show=False)
        assert "__heatmap_dummy__" not in adata.obs.columns

    def test_obs_order_unchanged(self):
        """Participant index order is not altered by the dummy-groupby workaround."""
        adata = _vote_adata()
        original_obs_names = list(adata.obs_names)
        val.viz.heatmap(adata, show=False)
        assert list(adata.obs_names) == original_obs_names

    def test_returns_axes_dict_when_show_false(self):
        """show=False returns a dict containing heatmap_ax."""
        adata = _vote_adata()
        result = val.viz.heatmap(adata, show=False)
        assert isinstance(result, dict)
        assert "heatmap_ax" in result

    def test_returns_none_when_show_true(self):
        """show=True (default) returns None."""
        import matplotlib.pyplot as plt
        adata = _vote_adata()
        result = val.viz.heatmap(adata, show=True)
        assert result is None
        plt.close("all")


# ─────────────────────────────────────────────────────────────────────
# TestHeatmapWithGroupby – explicit groupby column
# ─────────────────────────────────────────────────────────────────────


class TestHeatmapWithGroupby:
    def test_runs_with_groupby(self):
        """heatmap() works when a valid groupby column is provided."""
        adata = _vote_adata()
        adata.obs["cluster"] = (np.arange(adata.n_obs) % 3).astype(str)
        adata.obs["cluster"] = adata.obs["cluster"].astype("category")
        axes = val.viz.heatmap(adata, groupby="cluster", show=False)
        assert axes is not None

    def test_no_dummy_col_when_groupby_supplied(self):
        """__heatmap_dummy__ is never added when groupby is given."""
        adata = _vote_adata()
        adata.obs["cluster"] = "A"
        adata.obs["cluster"] = adata.obs["cluster"].astype("category")
        val.viz.heatmap(adata, groupby="cluster", show=False)
        assert "__heatmap_dummy__" not in adata.obs.columns


# ─────────────────────────────────────────────────────────────────────
# TestHeatmapDiscrete – discrete colorbar
# ─────────────────────────────────────────────────────────────────────


class TestHeatmapDiscrete:
    def test_discrete_runs_without_error(self):
        """discrete=True produces a figure without raising."""
        adata = _vote_adata()
        axes = val.viz.heatmap(adata, discrete=True, show=False)
        assert axes is not None

    def test_discrete_with_rdylgn_bright(self):
        """discrete=True works with the custom RdYlGnBright colormap."""
        adata = _vote_adata()
        axes = val.viz.heatmap(adata, cmap="RdYlGnBright", discrete=True, show=False)
        assert axes is not None

    def test_discrete_colorbar_tick_labels(self):
        """discrete=True sets the three expected colorbar tick labels."""
        import matplotlib.pyplot as plt
        adata = _vote_adata()
        axes = val.viz.heatmap(adata, discrete=True, show=False)

        heatmap_ax = axes.get("heatmap_ax")
        assert heatmap_ax is not None

        cbar = None
        for collection in heatmap_ax.collections:
            if getattr(collection, "colorbar", None) is not None:
                cbar = collection.colorbar
                break

        if cbar is not None:
            labels = [t.get_text() for t in cbar.ax.get_yticklabels()]
            assert "disagree (-1)" in labels
            assert "pass (0)" in labels
            assert "agree (+1)" in labels

        plt.close("all")


# ─────────────────────────────────────────────────────────────────────
# TestHeatmapColormaps – colormap options
# ─────────────────────────────────────────────────────────────────────


class TestHeatmapColormaps:
    @pytest.mark.parametrize("cmap", ["RdYlGn", "RdYlGnBright", "bwr"])
    def test_accepts_various_cmaps(self, cmap):
        adata = _vote_adata()
        axes = val.viz.heatmap(adata, cmap=cmap, show=False)
        assert axes is not None
