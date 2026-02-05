"""Unit tests for valency_anndata.tools._polis2.

All three polismath_commentgraph helpers are mocked at the module level so
that the optional dependency is never imported during the test suite.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from valency_anndata.tools._polis2 import recipe_polis2_statements


# ─────────────────────────────────────────────────────────────────────
# helpers & fixtures
# ─────────────────────────────────────────────────────────────────────

N_STATEMENTS = 30
NUM_LAYERS = 4
EMBED_DIM = 384


def _adata_with_statements(n=N_STATEMENTS):
    """AnnData with dense X and var['content']."""
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.standard_normal((10, n)))
    adata.var["content"] = [f"statement {i}" for i in range(n)]
    return adata


@pytest.fixture
def mock_embed():
    """Patches _embed_statements; yields (mock, deterministic embedding)."""
    rng = np.random.default_rng(1)
    embed = rng.standard_normal((N_STATEMENTS, EMBED_DIM))
    with patch(
        "valency_anndata.tools._polis2._embed_statements", return_value=embed
    ) as m:
        yield m, embed


@pytest.fixture
def mock_umap():
    """Patches _project_umap; yields (mock, deterministic 2-d projection)."""
    rng = np.random.default_rng(2)
    proj = rng.standard_normal((N_STATEMENTS, 2))
    with patch(
        "valency_anndata.tools._polis2._project_umap", return_value=proj
    ) as m:
        yield m, proj


@pytest.fixture
def mock_clusters():
    """Patches _create_cluster_layers; yields (mock, list of label arrays).

    Each layer cycles through 0-based integer labels so that assertions on
    specific values are deterministic.  Layer *i* uses ``i+1`` distinct labels.
    """
    layers = [
        np.array([j % (i + 1) for j in range(N_STATEMENTS)]) for i in range(NUM_LAYERS)
    ]
    with patch(
        "valency_anndata.tools._polis2._create_cluster_layers", return_value=layers
    ) as m:
        yield m, layers


# ─────────────────────────────────────────────────────────────────────
# TestRecipePolis2StatementsOutputKeys
# ─────────────────────────────────────────────────────────────────────


class TestRecipePolis2StatementsOutputKeys:
    def test_all_output_keys_present(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)

        assert "X_text_embed" in adata.varm
        assert "X_umap_statements" in adata.varm
        assert "evoc_polis2" in adata.varm
        assert "evoc_polis2_top" in adata.var.columns


# ─────────────────────────────────────────────────────────────────────
# TestRecipePolis2StatementsShapes
# ─────────────────────────────────────────────────────────────────────


class TestRecipePolis2StatementsShapes:
    def test_embed_shape(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)
        assert adata.varm["X_text_embed"].shape == (N_STATEMENTS, EMBED_DIM)

    def test_umap_shape(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)
        assert adata.varm["X_umap_statements"].shape == (N_STATEMENTS, 2)

    def test_evoc_polis2_shape(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)
        assert adata.varm["evoc_polis2"].shape == (N_STATEMENTS, NUM_LAYERS)

    def test_evoc_polis2_top_length(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)
        assert len(adata.var["evoc_polis2_top"]) == N_STATEMENTS


# ─────────────────────────────────────────────────────────────────────
# TestRecipePolis2StatementsInplace
# ─────────────────────────────────────────────────────────────────────


class TestRecipePolis2StatementsInplace:
    def test_inplace_true_returns_none(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        result = recipe_polis2_statements(adata, inplace=True)
        assert result is None
        assert "X_text_embed" in adata.varm

    def test_inplace_false_returns_copy(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        result = recipe_polis2_statements(adata, inplace=False)
        assert result is not None
        assert "X_text_embed" in result.varm
        # original is untouched
        assert "X_text_embed" not in adata.varm


# ─────────────────────────────────────────────────────────────────────
# TestRecipePolis2StatementsContentRequired
# ─────────────────────────────────────────────────────────────────────


class TestRecipePolis2StatementsContentRequired:
    def test_missing_content_raises(self, mock_embed, mock_umap, mock_clusters):
        rng = np.random.default_rng(0)
        adata = AnnData(X=rng.standard_normal((5, 10)))
        # no var["content"] column
        with pytest.raises(KeyError):
            recipe_polis2_statements(adata)


# ─────────────────────────────────────────────────────────────────────
# TestRecipePolis2StatementsCallArgs
# ─────────────────────────────────────────────────────────────────────


class TestRecipePolis2StatementsCallArgs:
    def test_embed_called_with_content(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)

        m_embed, _ = mock_embed
        expected_texts = [f"statement {i}" for i in range(N_STATEMENTS)]
        m_embed.assert_called_once_with(expected_texts)

    def test_umap_called_with_embeddings(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)

        _, embed = mock_embed
        m_umap, _ = mock_umap
        np.testing.assert_array_equal(m_umap.call_args[0][0], embed)

    def test_clusters_called_with_embeddings(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)

        _, embed = mock_embed
        m_clusters, _ = mock_clusters
        np.testing.assert_array_equal(m_clusters.call_args[0][0], embed)


# ─────────────────────────────────────────────────────────────────────
# TestRecipePolis2StatementsEvocTop
# ─────────────────────────────────────────────────────────────────────


class TestRecipePolis2StatementsEvocTop:
    def test_evoc_top_is_categorical(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)
        assert isinstance(adata.var["evoc_polis2_top"].dtype, pd.CategoricalDtype)

    def test_evoc_top_matches_coarsest_layer(self, mock_embed, mock_umap, mock_clusters):
        adata = _adata_with_statements()
        recipe_polis2_statements(adata)

        _, layers = mock_clusters
        coarsest = layers[-1]
        np.testing.assert_array_equal(
            adata.var["evoc_polis2_top"].cat.codes.values,
            pd.Categorical(coarsest).codes,
        )
