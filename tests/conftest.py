import shutil
from pathlib import Path

import pytest

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir():
    """Read-only path to the synthetic fixture directory."""
    return FIXTURES_ROOT / "polis_synthetic"


@pytest.fixture
def synthetic_fixture_dir(tmp_path):
    """Copy of synthetic fixtures in a tmp directory; safe to mutate."""
    dest = tmp_path / "polis_synthetic"
    shutil.copytree(FIXTURES_ROOT / "polis_synthetic", dest)
    return dest


@pytest.fixture
def real_fixture_dir():
    """Read-only path to the real (downloaded) Polis CSV exports."""
    return FIXTURES_ROOT / "polis_real"
