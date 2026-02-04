"""Live network tests for valency_anndata.datasets.polis.

All tests here are marked ``@pytest.mark.live`` and are **skipped by default**.
Run them explicitly:

    uv run pytest -m live
"""

import pytest

from valency_anndata.datasets.polis import load


pytestmark = pytest.mark.live


class TestLoadLive:
    def test_load_report_url(self):
        adata = load("https://pol.is/report/r2dxjrdwef2ybx2w9n3ja")
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert adata.uns["source"]["kind"] == "report"
        assert adata.raw is not None

    def test_load_convo_url(self):
        adata = load("https://pol.is/34rdajkfxk")
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert adata.uns["source"]["kind"] == "api"
        assert adata.raw is not None

    def test_load_bare_report_id(self):
        adata = load("r2dxjrdwef2ybx2w9n3ja")
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert adata.uns["source"]["kind"] == "report"
        assert adata.raw is not None

    def test_load_bare_convo_id(self):
        adata = load("34rdajkfxk")
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert adata.uns["source"]["kind"] == "api"
        assert adata.raw is not None
