"""Live network tests for valency_anndata.datasets.polis.

All tests here are marked ``@pytest.mark.live`` and are **skipped by default**.
Run them explicitly:

    uv run pytest -m live
"""

import numpy as np
import pandas as pd
import pytest
from io import StringIO
from polis_client import PolisClient

from valency_anndata.datasets.polis import export_csv, load


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


REPORT_ID = "r4j3kccpn73khcmasfcw9"
REPORT_URL = f"https://pol.is/report/{REPORT_ID}"


class TestExportCsvLive:
    """Compare CSV export generated from API data against server CSV export."""

    @pytest.fixture(scope="class")
    def server_votes(self):
        client = PolisClient(base_url="https://pol.is")
        text = client.get_export_file(filename="votes.csv", report_id=REPORT_ID)
        return pd.read_csv(StringIO(text))

    @pytest.fixture(scope="class")
    def server_comments(self):
        client = PolisClient(base_url="https://pol.is")
        text = client.get_export_file(filename="comments.csv", report_id=REPORT_ID)
        return pd.read_csv(StringIO(text))

    @pytest.fixture(scope="class")
    def api_adata(self):
        # Load via report URL to resolve conversation_id, then reload via API
        report_adata = load(REPORT_URL, build_X=False, skip_cache=True)
        convo_id = report_adata.uns["source"]["conversation_id"]
        return load(convo_id, skip_cache=True)

    @pytest.fixture(scope="class")
    def generated_csvs(self, api_adata, tmp_path_factory):
        out = tmp_path_factory.mktemp("export")
        export_csv(api_adata, str(out))
        return {
            "votes": pd.read_csv(out / "votes.csv"),
            "comments": pd.read_csv(out / "comments.csv"),
        }

    def test_vote_signs_match(self, server_votes, generated_csvs):
        """API-generated vote signs should match server CSV export."""
        gen_votes = generated_csvs["votes"]
        merged = server_votes[["comment-id", "voter-id", "vote"]].merge(
            gen_votes[["comment-id", "voter-id", "vote"]],
            on=["comment-id", "voter-id"],
            suffixes=("_server", "_generated"),
        )
        assert len(merged) > 0, "No overlapping votes found"
        mismatched = merged[merged["vote_server"] != merged["vote_generated"]]
        assert len(mismatched) == 0, (
            f"{len(mismatched)}/{len(merged)} votes have mismatched signs"
        )

    def test_vote_timestamps_are_seconds(self, generated_csvs):
        """Timestamps should be in seconds, not milliseconds."""
        ts = generated_csvs["votes"]["timestamp"]
        assert (ts < 1e12).all(), "Timestamps appear to be in milliseconds"

    def test_comments_columns_present(self, generated_csvs):
        """Generated comments.csv should have all expected columns."""
        expected = [
            "timestamp", "datetime", "comment-id", "author-id",
            "agrees", "disagrees", "moderated", "comment-body",
            "is-seed", "is-meta",
        ]
        actual = list(generated_csvs["comments"].columns)
        assert actual == expected

    def test_comments_timestamps_match(self, server_comments, generated_csvs):
        """Comment timestamps should match the server export."""
        gen_comments = generated_csvs["comments"]
        merged = server_comments[["comment-id", "timestamp"]].merge(
            gen_comments[["comment-id", "timestamp"]],
            on="comment-id",
            suffixes=("_server", "_generated"),
        )
        assert len(merged) > 0
        assert (merged["timestamp_server"] == merged["timestamp_generated"]).all()

    def test_comments_agrees_disagrees_reasonable(self, server_comments, generated_csvs):
        """Agrees/disagrees computed from matrix should be in the right ballpark."""
        gen = generated_csvs["comments"]
        merged = server_comments[["comment-id", "agrees", "disagrees"]].merge(
            gen[["comment-id", "agrees", "disagrees"]],
            on="comment-id",
            suffixes=("_server", "_generated"),
        )
        # Correlation should be high even if counts differ slightly
        # (server counts all votes; we count deduplicated last-votes)
        agrees_corr = np.corrcoef(
            merged["agrees_server"], merged["agrees_generated"]
        )[0, 1]
        disagrees_corr = np.corrcoef(
            merged["disagrees_server"], merged["disagrees_generated"]
        )[0, 1]
        assert agrees_corr > 0.95, f"Agrees correlation too low: {agrees_corr:.3f}"
        assert disagrees_corr > 0.95, f"Disagrees correlation too low: {disagrees_corr:.3f}"
