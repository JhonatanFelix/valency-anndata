"""Tests for val.preprocessing.impute()."""

import math
import numpy as np
import pytest
import anndata as ad

from valency_anndata.preprocessing import impute


def make_adata(X):
    return ad.AnnData(X=np.array(X, dtype=float))


class TestImputeZero:
    def test_replaces_nan_with_zero(self):
        adata = make_adata([[1.0, np.nan], [np.nan, -1.0]])
        impute(adata, strategy="zero")
        result = adata.layers["X_imputed_zero"]
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.0

    def test_non_nan_values_unchanged(self):
        adata = make_adata([[1.0, np.nan], [np.nan, -1.0]])
        impute(adata, strategy="zero")
        result = adata.layers["X_imputed_zero"]
        assert result[0, 0] == 1.0
        assert result[1, 1] == -1.0

    def test_no_nans_passes_through(self):
        adata = make_adata([[1.0, -1.0], [1.0, -1.0]])
        impute(adata, strategy="zero")
        result = adata.layers["X_imputed_zero"]
        np.testing.assert_array_equal(result, adata.X)


class TestImputeMean:
    def test_replaces_nan_with_column_mean(self):
        # col 0: values 1.0 and -1.0 → mean 0.0
        adata = make_adata([[1.0, 1.0], [np.nan, -1.0], [-1.0, 1.0]])
        impute(adata, strategy="mean")
        result = adata.layers["X_imputed_mean"]
        assert result[1, 0] == pytest.approx(0.0)

    def test_non_nan_values_unchanged(self):
        adata = make_adata([[1.0, np.nan], [-1.0, -1.0]])
        impute(adata, strategy="mean")
        result = adata.layers["X_imputed_mean"]
        assert result[0, 0] == 1.0
        assert result[1, 0] == -1.0
        assert result[1, 1] == -1.0


class TestImputeMedian:
    def test_replaces_nan_with_column_median(self):
        # col 0: observed values 1.0, -1.0, 1.0 → median 1.0
        adata = make_adata([[1.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [np.nan, -1.0]])
        impute(adata, strategy="median")
        result = adata.layers["X_imputed_median"]
        assert result[3, 0] == pytest.approx(1.0)

    def test_non_nan_values_unchanged(self):
        adata = make_adata([[1.0, np.nan], [-1.0, -1.0]])
        impute(adata, strategy="median")
        result = adata.layers["X_imputed_median"]
        assert result[0, 0] == 1.0
        assert result[1, 0] == -1.0


class TestImputeLayerHandling:
    def test_default_target_layer_name(self):
        adata = make_adata([[1.0, np.nan]])
        impute(adata, strategy="mean")
        assert "X_imputed_mean" in adata.layers

    def test_custom_target_layer(self):
        adata = make_adata([[1.0, np.nan]])
        impute(adata, strategy="mean", target_layer="my_layer")
        assert "my_layer" in adata.layers

    def test_skips_if_layer_exists(self):
        adata = make_adata([[1.0, np.nan]])
        adata.layers["X_imputed_mean"] = np.array([[99.0, 99.0]])
        impute(adata, strategy="mean")
        # Should not overwrite
        assert adata.layers["X_imputed_mean"][0, 0] == 99.0

    def test_overwrites_if_requested(self):
        adata = make_adata([[1.0, np.nan]])
        adata.layers["X_imputed_mean"] = np.array([[99.0, 99.0]])
        impute(adata, strategy="mean", overwrite=True)
        assert adata.layers["X_imputed_mean"][0, 0] == 1.0

    def test_source_layer(self):
        adata = make_adata([[1.0, 1.0]])
        adata.layers["masked"] = np.array([[np.nan, 1.0]])
        impute(adata, strategy="zero", source_layer="masked", target_layer="masked_imputed")
        assert adata.layers["masked_imputed"][0, 0] == 0.0
        assert adata.layers["masked_imputed"][0, 1] == 1.0

    def test_no_source_matrix_raises(self):
        adata = ad.AnnData()
        with pytest.raises(ValueError, match="No source matrix"):
            impute(adata, strategy="mean")


class TestImputeKNN:
    def test_imputes_nan_values(self):
        # Row 0 and row 2 are identical neighbors for row 1 (missing col 1)
        adata = make_adata([
            [1.0, 1.0],
            [1.0, np.nan],
            [1.0, 1.0],
        ])
        impute(adata, strategy="knn", params={"n_neighbors": 2})
        result = adata.layers["X_imputed_knn"]
        assert not np.isnan(result).any()
        assert result[1, 1] == pytest.approx(1.0)

    def test_default_target_layer_name(self):
        adata = make_adata([[1.0, np.nan], [-1.0, 1.0]])
        impute(adata, strategy="knn")
        assert "X_imputed_knn" in adata.layers

    def test_n_neighbors_via_params(self):
        adata = make_adata([
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, np.nan],
        ])
        impute(adata, strategy="knn", params={"n_neighbors": 1}, target_layer="knn_1")
        impute(adata, strategy="knn", params={"n_neighbors": 2}, target_layer="knn_2")
        # Both should produce valid (non-NaN) results
        assert not np.isnan(adata.layers["knn_1"]).any()
        assert not np.isnan(adata.layers["knn_2"]).any()

    def test_non_nan_values_unchanged(self):
        adata = make_adata([[1.0, 1.0], [-1.0, np.nan], [-1.0, -1.0]])
        impute(adata, strategy="knn", params={"n_neighbors": 2})
        result = adata.layers["X_imputed_knn"]
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(1.0)
        assert result[1, 0] == pytest.approx(-1.0)
        assert result[2, 0] == pytest.approx(-1.0)
        assert result[2, 1] == pytest.approx(-1.0)


class TestImputeUnknownStrategy:
    def test_raises_on_unknown_strategy(self):
        adata = make_adata([[1.0, np.nan]])
        with pytest.raises(ValueError, match="Unknown imputation strategy"):
            impute(adata, strategy="bogus")  # type: ignore[arg-type]
