from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ai_agent.utils import previews
from ai_agent.utils import cache_db
from ai_agent.utils.cache_db import CacheDB, reset_cache_db


@pytest.fixture(autouse=True)
def _isolated_cache_db():
    """Give every test a fresh in-memory SQLite cache."""
    db = CacheDB(":memory:")
    reset_cache_db(db)
    yield
    reset_cache_db(None)


def test_preview_cache_roundtrip_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key = ("/tmp/example.png::1::1",)
    preview_file = tmp_path / "preview.png"
    preview_file.write_bytes(b"x")

    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 30)

    previews._preview_cache_set(key, str(preview_file), "meta")
    out_path, out_meta = previews._preview_cache_get(key)

    assert out_path == str(preview_file)
    assert out_meta == "meta"


def test_preview_cache_evicts_expired_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    key = ("/tmp/expired.png::1::1",)
    preview_file = tmp_path / "expired.png"
    preview_file.write_bytes(b"x")

    now = {"t": 1_000_000.0}
    monkeypatch.setattr(cache_db.time, "time", lambda: now["t"])
    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 5)

    previews._preview_cache_set(key, str(preview_file), "meta")
    now["t"] = 1_000_006.0  # past TTL

    out_path, out_meta = previews._preview_cache_get(key)

    assert out_path is None
    assert out_meta is None


def test_preview_cache_drops_missing_file_entry(monkeypatch: pytest.MonkeyPatch):
    key = ("/tmp/missing.png::1::1",)

    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 3000)

    # Manually insert an entry pointing to a non-existent file
    db_key = json.dumps(key)
    value = json.dumps({"path": "/tmp/does-not-exist.png", "meta": "meta"})
    cache_db.get_cache_db().set(
        previews._PREVIEW_NS, db_key, value, ttl_seconds=3000
    )

    out_path, out_meta = previews._preview_cache_get(key)

    assert out_path is None
    assert out_meta is None


def test_preview_cache_capacity_eviction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 100)
    monkeypatch.setattr(previews, "PREVIEW_CACHE_MAX_ENTRIES", 2)

    now = {"t": 1_000_000.0}
    monkeypatch.setattr(cache_db.time, "time", lambda: now["t"])

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    c.write_bytes(b"c")

    key_a = ("a",)
    key_b = ("b",)
    key_c = ("c",)

    now["t"] = 1_000_001.0
    previews._preview_cache_set(key_a, str(a), "a")
    now["t"] = 1_000_002.0
    previews._preview_cache_set(key_b, str(b), "b")
    now["t"] = 1_000_003.0
    previews._preview_cache_set(key_c, str(c), "c")

    # key_a should have been evicted (oldest-accessed, capacity=2)
    assert previews._preview_cache_get(key_a) == (None, None)
    assert previews._preview_cache_get(key_b)[0] == str(b)
    assert previews._preview_cache_get(key_c)[0] == str(c)


def test_preview_cache_clear_helper_empties_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 30)

    preview_file = tmp_path / "x.png"
    preview_file.write_bytes(b"x")
    previews._preview_cache_set(("x",), str(preview_file), "meta")

    previews._clear_preview_cache_for_tests()

    assert previews._preview_cache_get(("x",)) == (None, None)


def test_resize_for_preview_preserves_aspect_ratio():
    img = Image.new("L", (2000, 1000), color=128)

    resized = previews._resize_for_preview(img, max_side_px=500)

    assert resized.size == (500, 250)


def test_resize_for_preview_does_not_upscale():
    img = Image.new("RGB", (320, 200), color=(0, 0, 0))

    resized = previews._resize_for_preview(img, max_side_px=500)

    assert resized.size == (320, 200)