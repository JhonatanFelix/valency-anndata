"""Unit and integration tests for valency_anndata.tools._kmeans.

Unit tests mock ``BestPolisKMeans`` at its import site so that no real
clustering is performed.  The single integration test at the bottom uses
the real class on a synthetically well-separated dataset.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData
from sklearn.metrics import silhouette_score

from valency_anndata.tools._kmeans import kmeans


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────


def _adata(n_obs=8, n_vars=3, obsm=None):
    """Minimal dense AnnData for unit tests."""
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.standard_normal((n_obs, n_vars)))
    if obsm:
        for k, v in obsm.items():
            adata.obsm[k] = v
    return adata


@pytest.fixture
def mock_bpk():
    """Patch BestPolisKMeans; yields (MockClass, mock_instance)."""
    with patch("valency_anndata.tools._kmeans.BestPolisKMeans") as MockClass:
        inst = MockClass.return_value
        inst.best_estimator_ = MagicMock()
        inst.best_estimator_.labels_ = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        inst.best_k_ = 2
        inst.best_score_ = 0.75
        yield MockClass, inst


# ─────────────────────────────────────────────────────────────────────
# TestKmeansRepresentation – _choose_representation paths
# ─────────────────────────────────────────────────────────────────────


class TestKmeansRepresentation:
    def test_uses_X_when_no_pca(self, mock_bpk):
        """When no X_pca exists, .X (8×3) is passed to .fit()."""
        _, inst = mock_bpk
        ad = _adata(n_obs=8, n_vars=3)
        kmeans(ad)
        fitted = inst.fit.call_args[0][0]
        assert fitted.shape == (8, 3)

    def test_uses_obsm_rep(self, mock_bpk):
        """use_rep='X_pca' routes to obsm['X_pca']."""
        _, inst = mock_bpk
        pca = np.ones((8, 5))
        ad = _adata(n_obs=8, obsm={"X_pca": pca})
        kmeans(ad, use_rep="X_pca")
        fitted = inst.fit.call_args[0][0]
        assert fitted.shape == (8, 5)

    def test_n_pcs_slices_columns(self, mock_bpk):
        """n_pcs=2 slices obsm down to 2 columns."""
        _, inst = mock_bpk
        pca = np.ones((8, 5))
        ad = _adata(n_obs=8, obsm={"X_pca": pca})
        kmeans(ad, use_rep="X_pca", n_pcs=2)
        fitted = inst.fit.call_args[0][0]
        assert fitted.shape == (8, 2)


# ─────────────────────────────────────────────────────────────────────
# TestKmeansValidation – error paths
# ─────────────────────────────────────────────────────────────────────


class TestKmeansValidation:
    def test_raises_on_sparse_X(self, mock_bpk):
        """Sparse .X (no obsm) triggers ValueError."""
        ad = AnnData(X=sp.csr_matrix(np.eye(8)))
        with pytest.raises(ValueError, match="numpy array"):
            kmeans(ad)

    def test_raises_when_no_valid_estimator(self, mock_bpk):
        """Falsy best_estimator_ triggers RuntimeError."""
        _, inst = mock_bpk
        inst.best_estimator_ = None
        ad = _adata()
        with pytest.raises(RuntimeError, match="did not find a valid estimator"):
            kmeans(ad)


# ─────────────────────────────────────────────────────────────────────
# TestKmeansKBounds – k_bounds normalisation
# ─────────────────────────────────────────────────────────────────────


class TestKmeansKBounds:
    def test_default_k_bounds(self, mock_bpk):
        """None → [2, 5] forwarded to constructor and stored in uns."""
        MockClass, _ = mock_bpk
        ad = _adata()
        kmeans(ad)
        assert MockClass.call_args[1]["k_bounds"] == [2, 5]
        assert ad.uns["kmeans"]["params"]["k_bounds"] == [2, 5]

    def test_explicit_k_bounds_converted_to_list(self, mock_bpk):
        """Tuple (3, 7) is converted to list [3, 7]."""
        MockClass, _ = mock_bpk
        ad = _adata()
        kmeans(ad, k_bounds=(3, 7))
        assert MockClass.call_args[1]["k_bounds"] == [3, 7]
        assert ad.uns["kmeans"]["params"]["k_bounds"] == [3, 7]


# ─────────────────────────────────────────────────────────────────────
# TestKmeansMaskObs – mask_obs sub-paths
# ─────────────────────────────────────────────────────────────────────


class TestKmeansMaskObs:
    def test_no_mask(self, mock_bpk):
        """mask_obs=None → all 8 obs clustered, no NaNs in labels."""
        _, inst = mock_bpk
        ad = _adata()
        kmeans(ad)
        assert ad.obs["kmeans"].isna().sum() == 0
        assert len(ad.obs["kmeans"]) == 8

    def test_bool_array_mask(self, mock_bpk):
        """Boolean array mask: only masked obs get labels; rest are NaN."""
        _, inst = mock_bpk
        mask = np.array([True, True, True, True, False, False, False, False])
        # 4 obs will be clustered → need 4 labels
        inst.best_estimator_.labels_ = np.array([0, 1, 0, 1])
        ad = _adata()
        kmeans(ad, mask_obs=mask)
        labels = ad.obs["kmeans"]
        # first 4 are assigned
        assert labels.iloc[:4].isna().sum() == 0
        # last 4 are NaN
        assert labels.iloc[4:].isna().sum() == 4

    def test_string_mask_from_obs(self, mock_bpk):
        """String mask_obs reads a boolean column from .obs."""
        _, inst = mock_bpk
        ad = _adata()
        ad.obs["use"] = [True, True, True, True, False, False, False, False]
        inst.best_estimator_.labels_ = np.array([0, 1, 0, 1])
        kmeans(ad, mask_obs="use")
        labels = ad.obs["kmeans"]
        assert labels.iloc[:4].isna().sum() == 0
        assert labels.iloc[4:].isna().sum() == 4

    def test_all_false_mask_raises(self, mock_bpk):
        """All-False boolean mask raises ValueError."""
        ad = _adata()
        mask = np.zeros(8, dtype=bool)
        with pytest.raises(ValueError, match="excludes all observations"):
            kmeans(ad, mask_obs=mask)

    def test_all_true_mask(self, mock_bpk):
        """All-True mask behaves like no mask — no NaNs."""
        _, inst = mock_bpk
        ad = _adata()
        mask = np.ones(8, dtype=bool)
        kmeans(ad, mask_obs=mask)
        assert ad.obs["kmeans"].isna().sum() == 0


# ─────────────────────────────────────────────────────────────────────
# TestKmeansInplace – return-value contract
# ─────────────────────────────────────────────────────────────────────


class TestKmeansInplace:
    def test_inplace_true(self, mock_bpk):
        """inplace=True returns None and mutates adata."""
        ad = _adata()
        result = kmeans(ad, inplace=True)
        assert result is None
        assert "kmeans" in ad.obs.columns

    def test_inplace_false_isolates_original(self, mock_bpk):
        """inplace=False returns a new AnnData; original is untouched."""
        ad = _adata()
        result = kmeans(ad, inplace=False)
        assert result is not None
        assert "kmeans" in result.obs.columns
        assert "kmeans" not in ad.obs.columns

    def test_inplace_false_with_mask(self, mock_bpk):
        """inplace=False + mask: returned copy has NaNs, original clean."""
        _, inst = mock_bpk
        mask = np.array([True, True, True, True, False, False, False, False])
        inst.best_estimator_.labels_ = np.array([0, 1, 0, 1])
        ad = _adata()
        result = kmeans(ad, mask_obs=mask, inplace=False)
        assert result.obs["kmeans"].iloc[4:].isna().sum() == 4
        assert "kmeans" not in ad.obs.columns


# ─────────────────────────────────────────────────────────────────────
# TestKmeansUnsParams – uns metadata completeness
# ─────────────────────────────────────────────────────────────────────


class TestKmeansUnsParams:
    def test_params_keys_and_values(self, mock_bpk):
        """All 7 expected keys present; values match inputs and mock."""
        ad = _adata()
        kmeans(ad, k_bounds=(2, 4), init="random", random_state=42,
               use_rep=None, n_pcs=None)
        params = ad.uns["kmeans"]["params"]
        assert set(params.keys()) == {
            "k_bounds", "best_k", "best_score", "init",
            "random_state", "use_rep", "n_pcs",
        }
        assert params["k_bounds"] == [2, 4]
        assert params["best_k"] == 2
        assert params["best_score"] == 0.75
        assert params["init"] == "random"
        assert params["random_state"] == 42
        assert params["use_rep"] is None
        assert params["n_pcs"] is None

    def test_params_defaults_when_optional_args_none(self, mock_bpk):
        """Calling with all defaults still records None placeholders."""
        ad = _adata()
        kmeans(ad)
        params = ad.uns["kmeans"]["params"]
        assert params["use_rep"] is None
        assert params["n_pcs"] is None
        assert params["random_state"] is None


# ─────────────────────────────────────────────────────────────────────
# TestKmeansKeyAdded – custom key_added
# ─────────────────────────────────────────────────────────────────────


class TestKmeansKeyAdded:
    def test_default_key_added(self, mock_bpk):
        """Default key is 'kmeans' in both obs and uns."""
        ad = _adata()
        kmeans(ad)
        assert "kmeans" in ad.obs.columns
        assert "kmeans" in ad.uns

    def test_custom_key_added(self, mock_bpk):
        """Custom key_added appears in both obs and uns."""
        ad = _adata()
        kmeans(ad, key_added="my_clusters")
        assert "my_clusters" in ad.obs.columns
        assert "my_clusters" in ad.uns
        # default key should NOT be present
        assert "kmeans" not in ad.obs.columns
        assert "kmeans" not in ad.uns


# ─────────────────────────────────────────────────────────────────────
# TestKmeansConstructorArgs – BestPolisKMeans instantiation
# ─────────────────────────────────────────────────────────────────────


class TestKmeansConstructorArgs:
    def test_init_and_random_state_forwarded(self, mock_bpk):
        """init and random_state are forwarded to BestPolisKMeans."""
        MockClass, _ = mock_bpk
        ad = _adata()
        kmeans(ad, init="random", random_state=7)
        kwargs = MockClass.call_args[1]
        assert kwargs["init"] == "random"
        assert kwargs["random_state"] == 7

    def test_init_centers_forwarded(self, mock_bpk):
        """init_centers array is forwarded verbatim."""
        MockClass, _ = mock_bpk
        centers = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        ad = _adata()
        kmeans(ad, init_centers=centers)
        kwargs = MockClass.call_args[1]
        np.testing.assert_array_equal(kwargs["init_centers"], centers)


# ─────────────────────────────────────────────────────────────────────
# TestKmeansKmeansPlusPlus – skipped until reddwarf supports kmeans++
# ─────────────────────────────────────────────────────────────────────

_SKIP_KMPLUSPLUS = pytest.mark.skip(
    reason="BestPolisKMeans does not yet support init='kmeans++'"
)


@_SKIP_KMPLUSPLUS
class TestKmeansKmeansPlusPlus:
    """Smoke tests for kmeans++ initialisation (real clustering, no mocks).

    These are skipped because ``reddwarf.sklearn.cluster.BestPolisKMeans``
    currently raises on ``init='kmeans++'``.  Un-skip once the upstream
    dependency adds support.
    """

    def test_kmeansplusplus_two_clusters(self):
        """kmeans++ finds two well-separated clusters."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(loc=-5, scale=0.3, size=(15, 2)),
            rng.normal(loc=+5, scale=0.3, size=(15, 2)),
        ])
        ad = AnnData(X=X)
        kmeans(ad, k_bounds=(2, 3), init="kmeans++", random_state=0)

        assert ad.uns["kmeans"]["params"]["best_k"] == 2
        labels = ad.obs["kmeans"].cat.codes.values
        assert len(set(labels[:15])) == 1
        assert len(set(labels[15:])) == 1
        assert set(labels[:15]) != set(labels[15:])

    def test_kmeansplusplus_params_recorded(self):
        """init='kmeans++' is faithfully stored in uns params."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(loc=-5, scale=0.3, size=(15, 2)),
            rng.normal(loc=+5, scale=0.3, size=(15, 2)),
        ])
        ad = AnnData(X=X)
        kmeans(ad, k_bounds=(2, 3), init="kmeans++", random_state=0)
        assert ad.uns["kmeans"]["params"]["init"] == "kmeans++"

    def test_kmeansplusplus_forwarded_to_constructor(self):
        """init='kmeans++' is passed through to BestPolisKMeans (mocked)."""
        with patch("valency_anndata.tools._kmeans.BestPolisKMeans") as MockClass:
            inst = MockClass.return_value
            inst.best_estimator_ = MagicMock()
            inst.best_estimator_.labels_ = np.array([0, 1, 0, 1, 0, 1, 0, 1])
            inst.best_k_ = 2
            inst.best_score_ = 0.75

            ad = _adata()
            kmeans(ad, init="kmeans++")
            assert MockClass.call_args[1]["init"] == "kmeans++"


# ─────────────────────────────────────────────────────────────────────
# TestKmeansIntegration – real BestPolisKMeans, no mocks
# ─────────────────────────────────────────────────────────────────────


class TestKmeansIntegration:
    def test_two_cluster_structure_detected(self):
        """Two well-separated clusters are correctly identified."""
        rng = np.random.default_rng(0)
        c1 = rng.normal(loc=-5, scale=0.3, size=(15, 2))
        c2 = rng.normal(loc=+5, scale=0.3, size=(15, 2))
        X = np.vstack([c1, c2])

        ad = AnnData(X=X)
        kmeans(ad, k_bounds=(2, 3), init="polis", random_state=0)

        params = ad.uns["kmeans"]["params"]
        assert params["best_k"] == 2

        labels = ad.obs["kmeans"].cat.codes.values
        # first 15 and last 15 must each be a single label
        first_labels = set(labels[:15])
        last_labels = set(labels[15:])
        assert len(first_labels) == 1
        assert len(last_labels) == 1
        assert first_labels != last_labels

        # silhouette score should be high for well-separated clusters
        score = silhouette_score(X, labels)
        assert 0 < score < 1
