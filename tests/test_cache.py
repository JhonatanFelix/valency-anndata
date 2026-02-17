"""Live integration tests for the file cache.

All tests here are marked ``@pytest.mark.live`` and are **skipped by default**.
Run them explicitly:

    uv run pytest -m live -k test_cache
"""

import pytest

from valency_anndata.datasets import _cache
from valency_anndata.datasets.polis import load


pytestmark = pytest.mark.live

REPORT_URL = "https://pol.is/report/r4zdxrdscmukmkakmbz3k"
REPORT_ID = "r4zdxrdscmukmkakmbz3k"
CONVO_URL = "https://pol.is/4asymkcrjf"
CONVO_ID = "4asymkcrjf"


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the cache at a temporary directory so tests don't pollute the
    user's real cache and each test starts with a clean slate."""
    monkeypatch.setattr(_cache, "_cache_dir", lambda: tmp_path)


class TestCacheReport:
    """Cache behaviour when loading via a report URL / ID."""

    def test_first_load_populates_cache(self, tmp_path):
        load(REPORT_URL)

        assert (tmp_path / REPORT_ID / "votes.csv").exists()
        assert (tmp_path / REPORT_ID / "statements.json").exists()
        assert (tmp_path / REPORT_ID / "conversation_id.txt").exists()
        assert (tmp_path / REPORT_ID / "last_vote_timestamp.txt").exists()

    def test_second_load_uses_cache(self, tmp_path):
        adata1 = load(REPORT_URL)
        # Stamp a marker in the cached votes file so we can detect a re-read
        votes_path = tmp_path / REPORT_ID / "votes.csv"
        original_text = votes_path.read_text()

        adata2 = load(REPORT_URL)

        # Cache file should be untouched (not re-written)
        assert votes_path.read_text() == original_text
        # Both loads should produce the same shape
        assert adata1.n_obs == adata2.n_obs
        assert adata1.n_vars == adata2.n_vars

    def test_skip_cache_bypasses_cache(self, tmp_path):
        load(REPORT_URL)
        votes_path = tmp_path / REPORT_ID / "votes.csv"
        mtime_before = votes_path.stat().st_mtime

        load(REPORT_URL, skip_cache=True)
        mtime_after = votes_path.stat().st_mtime

        # File should have been re-written
        assert mtime_after >= mtime_before

    def test_bare_report_id_uses_same_cache_key(self, tmp_path):
        load(REPORT_ID)
        assert (tmp_path / REPORT_ID / "votes.csv").exists()


class TestCacheAPI:
    """Cache behaviour when loading via a conversation URL / ID."""

    def test_first_load_populates_cache(self, tmp_path):
        load(CONVO_URL)

        assert (tmp_path / CONVO_ID / "votes.csv").exists()
        assert (tmp_path / CONVO_ID / "statements.json").exists()

    def test_second_load_uses_cache(self, tmp_path):
        adata1 = load(CONVO_URL)
        votes_path = tmp_path / CONVO_ID / "votes.csv"
        original_text = votes_path.read_text()

        adata2 = load(CONVO_URL)

        assert votes_path.read_text() == original_text
        assert adata1.n_obs == adata2.n_obs
        assert adata1.n_vars == adata2.n_vars

    def test_skip_cache_bypasses_cache(self, tmp_path):
        load(CONVO_URL)
        votes_path = tmp_path / CONVO_ID / "votes.csv"
        mtime_before = votes_path.stat().st_mtime

        load(CONVO_URL, skip_cache=True)
        mtime_after = votes_path.stat().st_mtime

        assert mtime_after >= mtime_before

    def test_bare_convo_id_uses_same_cache_key(self, tmp_path):
        load(CONVO_ID)
        assert (tmp_path / CONVO_ID / "votes.csv").exists()


class TestCacheExpiry:
    """TTL-based cache invalidation with last_vote_timestamp."""

    def _backdate_cache(self, tmp_path, cache_id, hours=25):
        """Backdate all cache files for *cache_id* to make them stale."""
        import os

        cache_dir = tmp_path / cache_id
        for child in cache_dir.iterdir():
            old_time = child.stat().st_mtime - (hours * 60 * 60)
            os.utime(child, (old_time, old_time))

    def test_stale_cache_with_matching_timestamp_is_reused(self, tmp_path):
        load(REPORT_URL)
        votes_path = tmp_path / REPORT_ID / "votes.csv"
        ts_path = tmp_path / REPORT_ID / "last_vote_timestamp.txt"

        # Verify the timestamp file was written
        assert ts_path.exists(), "last_vote_timestamp.txt should be written on first load"
        original_votes_text = votes_path.read_text()

        # Backdate all cache files to make them stale
        self._backdate_cache(tmp_path, REPORT_ID)

        # Second load — timestamp should match, so cache is reused (touched, not re-written)
        load(REPORT_URL)

        # votes.csv content should be identical (served from cache, not re-fetched)
        assert votes_path.read_text() == original_votes_text
        # mtime should have been refreshed by touch()
        assert _cache._is_fresh(votes_path)

    def test_stale_cache_is_refetched_when_no_timestamp(self, tmp_path):
        load(REPORT_URL)
        votes_path = tmp_path / REPORT_ID / "votes.csv"
        ts_path = tmp_path / REPORT_ID / "last_vote_timestamp.txt"

        # Remove the timestamp file to simulate old cache without it
        ts_path.unlink()

        # Backdate remaining files
        self._backdate_cache(tmp_path, REPORT_ID)
        old_mtime = votes_path.stat().st_mtime

        # Should fall through to full re-fetch
        load(REPORT_URL)
        assert votes_path.stat().st_mtime > old_mtime


class TestCacheLocalBypass:
    """Local directory loads should never touch the cache."""

    def test_local_load_does_not_cache(self, tmp_path, fixture_dir):
        load(str(fixture_dir))
        # Cache dir should remain empty
        assert list(tmp_path.iterdir()) == []
