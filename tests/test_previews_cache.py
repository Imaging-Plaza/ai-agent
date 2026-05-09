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


# ---------------------------------------------------------------------------
# Tests for resize_uploaded_image
# ---------------------------------------------------------------------------

def test_resize_uploaded_image_large_landscape(tmp_path: Path):
    """4000×3000 image → 500×375 (landscape, aspect ratio preserved)."""
    src = tmp_path / "big.png"
    Image.new("RGB", (4000, 3000), color=(255, 0, 0)).save(str(src))

    result = previews.resize_uploaded_image(str(src))

    assert result != str(src), "Should write to a new temp file"
    with Image.open(result) as out_img:
        assert out_img.size == (500, 375)


def test_resize_uploaded_image_large_portrait(tmp_path: Path):
    """3000×4000 image → 375×500 (portrait, aspect ratio preserved)."""
    src = tmp_path / "portrait.png"
    Image.new("RGB", (3000, 4000), color=(0, 255, 0)).save(str(src))

    result = previews.resize_uploaded_image(str(src))

    assert result != str(src)
    with Image.open(result) as out_img:
        assert out_img.size == (375, 500)


def test_resize_uploaded_image_already_small_unchanged(tmp_path: Path):
    """Images within bounds are returned as-is (no temp copy created)."""
    src = tmp_path / "small.png"
    Image.new("RGB", (200, 100), color=(0, 0, 255)).save(str(src))

    result = previews.resize_uploaded_image(str(src))

    assert result == str(src), "Small images should not be resized"


def test_resize_uploaded_image_exact_bound_unchanged(tmp_path: Path):
    """A 500×500 image is exactly at the limit and should not be resized."""
    src = tmp_path / "exact.png"
    Image.new("RGB", (500, 500), color=(128, 128, 128)).save(str(src))

    result = previews.resize_uploaded_image(str(src))

    assert result == str(src)


def test_resize_uploaded_image_non_image_passthrough(tmp_path: Path):
    """Non-image files (e.g. DICOM, CSV) are returned unchanged."""
    dcm = tmp_path / "scan.dcm"
    dcm.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 100)

    result = previews.resize_uploaded_image(str(dcm))

    assert result == str(dcm)


def test_resize_uploaded_image_jpeg_no_transparency(tmp_path: Path):
    """JPEG output must be RGB (no alpha channel)."""
    src = tmp_path / "rgba.jpg"
    # Create a large RGBA PNG but save as JPEG (which strips alpha anyway via convert)
    img = Image.new("RGBA", (2000, 1500), color=(10, 20, 30, 128))
    img.convert("RGB").save(str(src), format="JPEG")

    result = previews.resize_uploaded_image(str(src))

    assert result != str(src)
    with Image.open(result) as out_img:
        assert out_img.mode == "RGB"
        assert out_img.size == (500, 375)


def test_resize_uploaded_image_custom_bounds(tmp_path: Path):
    """Custom max_width / max_height are respected."""
    src = tmp_path / "img.png"
    Image.new("RGB", (1000, 500), color=(0, 0, 0)).save(str(src))

    result = previews.resize_uploaded_image(str(src), max_width=200, max_height=200)

    with Image.open(result) as out_img:
        # 1000×500 scaled to fit 200×200: width limited → 200×100
        assert out_img.size == (200, 100)