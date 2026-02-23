"""Regression tests: no warnings from public API methods on real data.

Two complementary sets of tests per method:

  TestNoRuntimeWarnings* — strict: RuntimeWarning is always an error.
    These catch numerical issues (division by zero, invalid values).
    All currently pass. Failing one is a regression to fix immediately.

  TestNoAnyWarnings* — aspirational: all warning categories are errors.
    These catch UserWarning, FutureWarning, DeprecationWarning, etc. from
    third-party deps. Methods that currently emit any warning are marked
    xfail so they appear in the report but don't block CI. Resolve the
    underlying issue and remove the xfail marker when clean.

Methods intentionally omitted:
  - Scanpy re-exports (pca, tsne, umap, leiden, neighbors): third-party code
  - val.tl.recipe_polis2_statements: requires optional polismath-commentgraph dep
  - val.viz.langevitour, .voter_vignette_browser, .jscatter: require notebook env
"""
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

import valency_anndata as val

_REAL_FIXTURE = Path(__file__).parent / "fixtures" / "polis_real"


@contextmanager
def _assert_no_runtime_warnings():
    """Fail if any RuntimeWarning is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


@contextmanager
def _assert_no_any_warnings():
    """Fail if any warning of any category is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ─────────────────────────────────────────────────────────────────────
# Shared fixtures (module-scoped so real data is loaded once per session)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_adata():
    """Real Polis conversation loaded from local CSV fixtures (no network)."""
    return val.datasets.load(str(_REAL_FIXTURE))


@pytest.fixture(scope="module")
def imputed_adata(real_adata):
    """adata with imputation applied — required by pacmap and localmap."""
    adata = real_adata.copy()
    val.pp.impute(adata)
    return adata


@pytest.fixture(scope="module")
def pipeline_adata(real_adata):
    """adata after recipe_polis — needed by downstream methods (kmeans).

    Raw Polis CSV exports omit the is_meta column; fill with False (no
    meta-statements) so that recipe_polis's zero-mask step can proceed.
    """
    adata = real_adata.copy()
    adata.var["is_meta"] = adata.var["is_meta"].eq(True)  # NaN → False
    val.tl.recipe_polis(adata)
    return adata


@pytest.fixture(scope="module")
def hvs_adata(real_adata):
    """adata after highly_variable_statements — needed by val.viz.highly_variable_statements."""
    adata = real_adata.copy()
    val.pp.highly_variable_statements(adata, n_top_statements=20)
    return adata


# ─────────────────────────────────────────────────────────────────────
# val.preprocessing
# ─────────────────────────────────────────────────────────────────────


class TestNoRuntimeWarningsPreprocessing:
    def test_rebuild_vote_matrix(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_runtime_warnings():
            val.pp.rebuild_vote_matrix(adata)

    def test_impute(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_runtime_warnings():
            val.pp.impute(adata)

    def test_calculate_qc_metrics(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_runtime_warnings():
            val.pp.calculate_qc_metrics(adata)

    def test_highly_variable_statements(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_runtime_warnings():
            val.pp.highly_variable_statements(adata)


# ─────────────────────────────────────────────────────────────────────
# val.tools
# ─────────────────────────────────────────────────────────────────────


class TestNoRuntimeWarningsTools:
    def test_recipe_polis(self, real_adata):
        adata = real_adata.copy()
        # Raw Polis CSV exports omit is_meta; fill with False to satisfy zero-mask.
        adata.var["is_meta"] = adata.var["is_meta"].eq(True)  # NaN → False
        with _assert_no_runtime_warnings():
            val.tl.recipe_polis(adata)

    def test_kmeans(self, pipeline_adata):
        # use_rep must be explicit: default falls back to adata.X which has NaN.
        adata = pipeline_adata.copy()
        with _assert_no_runtime_warnings():
            val.tl.kmeans(adata, use_rep="X_pca_polis")

    def test_pacmap(self, imputed_adata):
        # pacmap defaults to layer='X_imputed'; impute() writes 'X_imputed_mean'.
        adata = imputed_adata.copy()
        with _assert_no_runtime_warnings():
            val.tl.pacmap(adata, layer="X_imputed_mean")

    def test_localmap(self, imputed_adata):
        # localmap defaults to layer='X_imputed'; impute() writes 'X_imputed_mean'.
        adata = imputed_adata.copy()
        with _assert_no_runtime_warnings():
            val.tl.localmap(adata, layer="X_imputed_mean")


# ─────────────────────────────────────────────────────────────────────
# val.viz
# ─────────────────────────────────────────────────────────────────────


class TestNoRuntimeWarningsViz:
    def test_schematic_diagram(self, real_adata):
        with patch("webbrowser.open"), patch("webbrowser.get"):
            with _assert_no_runtime_warnings():
                val.viz.schematic_diagram(real_adata)

    def test_highly_variable_statements(self, hvs_adata):
        with patch("valency_anndata.viz._highly_variable_statements.plt"), \
             patch("valency_anndata.viz._highly_variable_statements.savefig_or_show"):
            with _assert_no_runtime_warnings():
                val.viz.highly_variable_statements(hvs_adata)

    @pytest.mark.skip(reason="requires notebook/widget environment")
    def test_langevitour(self, real_adata): ...

    @pytest.mark.skip(reason="requires notebook/widget environment")
    def test_voter_vignette_browser(self, real_adata): ...

    @pytest.mark.skip(reason="requires notebook/widget environment")
    def test_jscatter(self, real_adata): ...


# ─────────────────────────────────────────────────────────────────────
# Aspirational: no warnings of any category (UserWarning, FutureWarning,
# DeprecationWarning, etc.).  Methods that currently emit third-party
# warnings are marked xfail — resolve and remove the marker when clean.
# ─────────────────────────────────────────────────────────────────────


class TestNoAnyWarningsPreprocessing:
    def test_rebuild_vote_matrix(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_any_warnings():
            val.pp.rebuild_vote_matrix(adata)

    def test_impute(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_any_warnings():
            val.pp.impute(adata)

    def test_calculate_qc_metrics(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_any_warnings():
            val.pp.calculate_qc_metrics(adata)

    def test_highly_variable_statements(self, real_adata):
        adata = real_adata.copy()
        with _assert_no_any_warnings():
            val.pp.highly_variable_statements(adata)


class TestNoAnyWarningsTools:
    def test_recipe_polis(self, real_adata):
        adata = real_adata.copy()
        adata.var["is_meta"] = adata.var["is_meta"].eq(True)  # NaN → False
        with _assert_no_any_warnings():
            val.tl.recipe_polis(adata)

    def test_kmeans(self, pipeline_adata):
        adata = pipeline_adata.copy()
        with _assert_no_any_warnings():
            val.tl.kmeans(adata, use_rep="X_pca_polis")

    def test_pacmap(self, imputed_adata):
        adata = imputed_adata.copy()
        with _assert_no_any_warnings():
            val.tl.pacmap(adata, layer="X_imputed_mean")

    def test_localmap(self, imputed_adata):
        adata = imputed_adata.copy()
        with _assert_no_any_warnings():
            val.tl.localmap(adata, layer="X_imputed_mean")


class TestNoAnyWarningsViz:
    def test_schematic_diagram(self, real_adata):
        with patch("webbrowser.open"), patch("webbrowser.get"):
            with _assert_no_any_warnings():
                val.viz.schematic_diagram(real_adata)

    def test_highly_variable_statements(self, hvs_adata):
        with patch("valency_anndata.viz._highly_variable_statements.plt"), \
             patch("valency_anndata.viz._highly_variable_statements.savefig_or_show"):
            with _assert_no_any_warnings():
                val.viz.highly_variable_statements(hvs_adata)

    @pytest.mark.skip(reason="requires notebook/widget environment")
    def test_langevitour(self, real_adata): ...

    @pytest.mark.skip(reason="requires notebook/widget environment")
    def test_voter_vignette_browser(self, real_adata): ...

    @pytest.mark.skip(reason="requires notebook/widget environment")
    def test_jscatter(self, real_adata): ...
