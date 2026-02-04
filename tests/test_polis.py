"""Unit and local-fixture integration tests for valency_anndata.datasets.polis.

No network access is required; all tests use checked-in CSV fixtures or
exercise pure-function logic.
"""

import math
import numpy as np
import pandas as pd
import pytest

from valency_anndata.datasets.polis import (
    PolisSource,
    _parse_polis_source,
    format_attribution,
    load,
)


# ─────────────────────────────────────────────────────────────────────
# _parse_polis_source – URL / ID / local-dir parsing
# ─────────────────────────────────────────────────────────────────────


class TestParsePolisSource:
    def test_parse_report_url(self):
        src = _parse_polis_source("https://pol.is/report/r2dxjrdwef2ybx2w9n3ja")
        assert src.kind == "report"
        assert src.base_url == "https://pol.is"
        assert src.report_id == "r2dxjrdwef2ybx2w9n3ja"
        assert src.conversation_id is None

    def test_parse_convo_url(self):
        src = _parse_polis_source("https://pol.is/34rdajkfxk")
        assert src.kind == "api"
        assert src.base_url == "https://pol.is"
        assert src.conversation_id == "34rdajkfxk"
        assert src.report_id is None

    def test_parse_bare_convo_id(self):
        src = _parse_polis_source("34rdajkfxk")
        assert src.kind == "api"
        assert src.base_url == "https://pol.is"
        assert src.conversation_id == "34rdajkfxk"

    def test_parse_bare_report_id(self):
        src = _parse_polis_source("r2dxjrdwef2ybx2w9n3ja")
        assert src.kind == "report"
        assert src.base_url == "https://pol.is"
        assert src.report_id == "r2dxjrdwef2ybx2w9n3ja"

    def test_parse_custom_host_report_url(self):
        src = _parse_polis_source("https://polis.tw/report/r2dxjrdwef2ybx2w9n3ja")
        assert src.kind == "report"
        assert src.base_url == "https://polis.tw"
        assert src.report_id == "r2dxjrdwef2ybx2w9n3ja"

    def test_parse_custom_host_convo_url(self):
        src = _parse_polis_source("https://polis.tw/34rdajkfxk")
        assert src.kind == "api"
        assert src.base_url == "https://polis.tw"
        assert src.conversation_id == "34rdajkfxk"

    def test_parse_local_dir(self, synthetic_fixture_dir):
        src = _parse_polis_source(str(synthetic_fixture_dir))
        assert src.kind == "local"
        assert src.path == synthetic_fixture_dir

    def test_parse_local_dir_missing_votes(self, synthetic_fixture_dir):
        (synthetic_fixture_dir / "votes.csv").unlink()
        with pytest.raises(ValueError, match="votes.csv"):
            _parse_polis_source(str(synthetic_fixture_dir))

    def test_parse_local_dir_missing_comments(self, synthetic_fixture_dir):
        (synthetic_fixture_dir / "comments.csv").unlink()
        with pytest.raises(ValueError, match="comments.csv"):
            _parse_polis_source(str(synthetic_fixture_dir))

    def test_parse_invalid_source(self):
        with pytest.raises(ValueError, match="Unrecognized Polis source"):
            _parse_polis_source("not-a-valid-source!!!")

    def test_parse_url_trailing_slash(self):
        src = _parse_polis_source("https://pol.is/report/r2dxjrdwef2ybx2w9n3ja/")
        assert src.kind == "report"
        assert src.report_id == "r2dxjrdwef2ybx2w9n3ja"

    def test_parse_url_whitespace(self):
        src = _parse_polis_source("  https://pol.is/34rdajkfxk  ")
        assert src.kind == "api"
        assert src.conversation_id == "34rdajkfxk"



# ─────────────────────────────────────────────────────────────────────
# format_attribution – text wrapping
# ─────────────────────────────────────────────────────────────────────


class TestFormatAttribution:
    def test_format_attribution_short(self):
        result = format_attribution("Short text")
        assert result == "Short text"

    def test_format_attribution_wraps(self):
        long_para = " ".join(["word"] * 40)  # ~200 chars
        result = format_attribution(long_para)
        for line in result.split("\n"):
            assert len(line) <= 80

    def test_format_attribution_two_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = format_attribution(text)
        # Two paragraphs joined by single \n (each paragraph is one line here)
        assert result == "First paragraph.\nSecond paragraph."

    def test_format_attribution_custom_width(self):
        text = " ".join(["word"] * 20)  # ~100 chars
        result = format_attribution(text, width=30)
        for line in result.split("\n"):
            assert len(line) <= 30

    def test_format_attribution_no_hyphen_break(self):
        # A hyphenated compound word should not be broken across lines
        text = "See https://compdemocracy.org/polis for details about the project."
        result = format_attribution(text, width=40)
        # The URL should appear intact on a single line
        assert "https://compdemocracy.org/polis" in result


# ─────────────────────────────────────────────────────────────────────
# load() – integration tests using local synthetic fixtures
# ─────────────────────────────────────────────────────────────────────


class TestLoadLocal:
    def test_load_local_shape_and_keys(self, fixture_dir):
        adata = load(str(fixture_dir))
        assert adata.X.shape == (5, 4)
        for key in ("votes", "votes_meta", "statements", "statements_meta", "source", "schema"):
            assert key in adata.uns
        assert adata.raw is not None
        assert "raw_sparse" in adata.layers

    def test_load_local_build_X_false(self, fixture_dir):
        adata = load(str(fixture_dir), build_X=False)
        assert adata.n_obs == 0
        assert adata.n_vars == 0
        assert adata.raw is None
        # uns is still populated
        assert "votes" in adata.uns
        assert "statements" in adata.uns

    def test_load_local_vote_matrix_values(self, fixture_dir):
        adata = load(str(fixture_dir))

        # obs index is stringified voter-ids; var index is stringified comment-ids
        obs_idx = list(adata.obs_names)
        var_idx = list(adata.var_names)

        def cell(voter_id, stmt_id):
            r = obs_idx.index(str(voter_id))
            c = var_idx.index(str(stmt_id))
            return adata.X[r, c]

        # Voter 101 voted on all 4 statements
        assert cell(101, 0) == 1.0
        assert cell(101, 1) == -1.0
        assert cell(101, 2) == 1.0
        assert cell(101, 3) == -1.0

        # Voter 103, stmt 1: duplicate rows → last timestamp (1700000600) wins → -1
        assert cell(103, 1) == -1.0

        # Voter 102, stmt 2: no vote → NaN
        assert math.isnan(cell(102, 2))

        # Voter 103, stmt 0: no vote → NaN
        assert math.isnan(cell(103, 0))

        # Voter 104, stmts 2 and 3: no vote → NaN
        assert math.isnan(cell(104, 2))
        assert math.isnan(cell(104, 3))

    def test_load_local_var_metadata(self, fixture_dir):
        adata = load(str(fixture_dir))

        # Statement 0 is the seed
        assert adata.var.loc["0", "content"] == "This is the seed statement"
        assert adata.var.loc["0", "is_seed"] == True  # noqa: E712

        # Statement 2 is meta
        assert adata.var.loc["2", "is_meta"] == True  # noqa: E712

        # Statement 3 was moderated out but still present (filtering is downstream)
        assert adata.var.loc["3", "moderation_state"] == -1

    def test_load_local_votes_uns_has_all_rows(self, fixture_dir):
        adata = load(str(fixture_dir))
        # uns["votes"] stores the raw rows before dedup — 16 rows in fixture
        assert len(adata.uns["votes"]) == 16

    def test_load_local_votes_meta_structure(self, fixture_dir):
        adata = load(str(fixture_dir))
        meta = adata.uns["votes_meta"]
        assert "local" in meta["sources"]
        assert meta["sources"]["local"]["via"] == "filesystem"
        assert "retrieved_at" in meta["sources"]["local"]



# ─────────────────────────────────────────────────────────────────────
# load() – real downloaded CSV export (local file, no network)
# ─────────────────────────────────────────────────────────────────────


class TestLoadRealCsvExport:
    def test_load_real_csv_export(self, real_fixture_dir):
        # The real comments.csv from this report does NOT include is-seed or
        # is-meta columns.  The code handles that gracefully (sets to pd.NA).
        adata = load(str(real_fixture_dir))

        # Non-empty matrix
        assert adata.n_obs > 0
        assert adata.n_vars > 0

        # All expected uns keys present
        for key in ("votes", "votes_meta", "statements", "statements_meta", "source", "schema"):
            assert key in adata.uns

        # var has the expected metadata columns
        assert "content" in adata.var.columns
        assert "moderation_state" in adata.var.columns

        # raw snapshot was taken
        assert adata.raw is not None
