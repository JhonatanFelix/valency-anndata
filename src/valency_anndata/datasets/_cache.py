"""Simple file-based cache with 24-hour TTL using platformdirs."""

import json
import time
from pathlib import Path

from platformdirs import user_cache_dir

APP_NAME = "valency-anndata"
TTL_SECONDS = 24 * 60 * 60  # 24 hours


def _cache_dir() -> Path:
    return Path(user_cache_dir(APP_NAME))


def _is_fresh(path: Path) -> bool:
    """Return True if *path* exists and was modified less than TTL_SECONDS ago."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < TTL_SECONDS


def get(key: str) -> str | None:
    """Return cached text for *key*, or None if missing / stale."""
    path = _cache_dir() / key
    if _is_fresh(path):
        return path.read_text()
    return None


def put(key: str, data: str) -> None:
    """Write *data* to the cache under *key*."""
    path = _cache_dir() / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


def get_json(key: str):
    """Return cached JSON-deserialised object, or None if missing / stale."""
    text = get(key)
    if text is not None:
        return json.loads(text)
    return None


def put_json(key: str, obj) -> None:
    """Serialise *obj* to JSON and cache it under *key*."""
    put(key, json.dumps(obj))
