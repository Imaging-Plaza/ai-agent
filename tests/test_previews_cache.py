from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ai_agent.utils import previews


@pytest.fixture(autouse=True)
def _clear_preview_cache_between_tests():
    previews._clear_preview_cache_for_tests()
    yield
    previews._clear_preview_cache_for_tests()


def test_preview_cache_roundtrip_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    key = ("/tmp/example.png::1::1",)
    preview_file = tmp_path / "preview.png"
    preview_file.write_bytes(b"x")

    now = {"t": 100.0}
    monkeypatch.setattr(previews.time, "monotonic", lambda: now["t"])
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

    now = {"t": 10.0}
    monkeypatch.setattr(previews.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 5)

    previews._preview_cache_set(key, str(preview_file), "meta")
    now["t"] = 16.0

    out_path, out_meta = previews._preview_cache_get(key)

    assert out_path is None
    assert out_meta is None
    assert key not in previews._PREVIEW_CACHE


def test_preview_cache_drops_missing_file_entry(monkeypatch: pytest.MonkeyPatch):
    key = ("/tmp/missing.png::1::1",)

    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 30)
    with previews._PREVIEW_CACHE_LOCK:
        previews._PREVIEW_CACHE[key] = (previews.time.monotonic() + 10, "/tmp/does-not-exist.png", "meta")

    out_path, out_meta = previews._preview_cache_get(key)

    assert out_path is None
    assert out_meta is None
    assert key not in previews._PREVIEW_CACHE


def test_preview_cache_capacity_eviction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 100)
    monkeypatch.setattr(previews, "PREVIEW_CACHE_MAX_ENTRIES", 2)

    t = {"v": 0.0}
    monkeypatch.setattr(previews.time, "monotonic", lambda: t["v"])

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    c.write_bytes(b"c")

    key_a = ("a",)
    key_b = ("b",)
    key_c = ("c",)

    t["v"] = 1.0
    previews._preview_cache_set(key_a, str(a), "a")
    t["v"] = 2.0
    previews._preview_cache_set(key_b, str(b), "b")
    t["v"] = 3.0
    previews._preview_cache_set(key_c, str(c), "c")

    assert key_a not in previews._PREVIEW_CACHE
    assert key_b in previews._PREVIEW_CACHE
    assert key_c in previews._PREVIEW_CACHE


def test_preview_cache_clear_helper_empties_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(previews, "PREVIEW_CACHE_TTL_SECONDS", 30)

    preview_file = tmp_path / "x.png"
    preview_file.write_bytes(b"x")
    previews._preview_cache_set(("x",), str(preview_file), "meta")

    previews._clear_preview_cache_for_tests()

    assert previews._PREVIEW_CACHE == {}


def test_resize_for_preview_preserves_aspect_ratio():
    img = Image.new("L", (2000, 1000), color=128)

    resized = previews._resize_for_preview(img, max_side_px=500)

    assert resized.size == (500, 250)


def test_resize_for_preview_does_not_upscale():
    img = Image.new("RGB", (320, 200), color=(0, 0, 0))

    resized = previews._resize_for_preview(img, max_side_px=500)

    assert resized.size == (320, 200)