"""On-demand asset views — slice / MIP / metadata.
Loads a session asset (TIFF / DICOM / NIfTI / regular image), extracts a 2D
view, optionally applies gamma + contrast, and encodes it as PNG.

Keeps the work CPU-only and stateless so it can be served straight from a
``GET /api/files/asset/{id}/view`` endpoint.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from PIL import Image

from ai_agent.services.sessions import Asset
from ai_agent.utils.image_io import load_any

log = logging.getLogger("services.views")

ViewKind = Literal["slice", "mip"]
Axis = Literal["z", "y", "x"]


MediaType = Literal["image", "volume", "video", "audio", "pdf", "other"]

VIDEO_EXTS = {"mp4", "mov", "webm", "mkv"}
AUDIO_EXTS = {"mp3", "wav", "ogg", "flac", "m4a"}
PDF_EXTS = {"pdf"}


def _detect_media_type(fmt: Optional[str], is_volume: bool, is_rgb: bool) -> MediaType:
    f = (fmt or "").lower()
    if f in VIDEO_EXTS:
        return "video"
    if f in AUDIO_EXTS:
        return "audio"
    if f in PDF_EXTS:
        return "pdf"
    if is_volume:
        return "volume"
    if is_rgb or f in {"png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp", "gif", "nii", "nii.gz", "dcm"}:
        return "image"
    return "other"


@dataclass
class AssetInfo:
    asset_id: str
    display_name: Optional[str]
    original_format: Optional[str]
    metadata_text: Optional[str]
    media_type: MediaType
    file_size: Optional[int]
    ndim: int
    shape: list[int]
    dtype: str
    intensity_min: float
    intensity_max: float
    is_rgb: bool
    is_volume: bool
    axes: Dict[str, Optional[int]]  # 'z' -> dim along axis 0 etc.
    extra: Dict[str, Any]


_volume_cache: dict[str, Tuple[float, np.ndarray, Dict[str, Any]]] = {}
_CACHE_TTL_S = 600


def _evict_expired() -> None:
    now = time.time()
    expired = [k for k, (t, _, _) in _volume_cache.items() if now - t > _CACHE_TTL_S]
    for k in expired:
        _volume_cache.pop(k, None)


def _load_asset_array(asset: Asset) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load an asset's underlying file once, then keep it warm for follow-ups."""
    _evict_expired()
    cached = _volume_cache.get(asset.asset_id)
    if cached:
        return cached[1], cached[2]
    arr, meta = load_any(asset.path)
    _volume_cache[asset.asset_id] = (time.time(), arr, meta)
    return arr, meta


def _is_rgb_shape(shape: Tuple[int, ...]) -> bool:
    return shape and shape[-1] in (3, 4) and len(shape) in (3, 4)


def get_info(asset: Asset) -> AssetInfo:
    fmt = (asset.original_format or "").lower()
    file_size: Optional[int] = None
    try:
        file_size = os.path.getsize(asset.path)
    except Exception:
        pass

    # Non-imagery media (video/audio/pdf): skip array loading entirely — those
    # files would crash load_any. We still return a populated AssetInfo so the
    # frontend can render the right tabs.
    if fmt in VIDEO_EXTS or fmt in AUDIO_EXTS or fmt in PDF_EXTS:
        media = _detect_media_type(fmt, is_volume=False, is_rgb=False)
        return AssetInfo(
            asset_id=asset.asset_id,
            display_name=asset.display_name,
            original_format=asset.original_format,
            metadata_text=asset.metadata_text,
            media_type=media,
            file_size=file_size,
            ndim=0,
            shape=[],
            dtype="binary",
            intensity_min=0.0,
            intensity_max=0.0,
            is_rgb=False,
            is_volume=False,
            axes={"z": None, "y": None, "x": None},
            extra={},
        )

    arr, meta = _load_asset_array(asset)
    shape = tuple(int(s) for s in arr.shape)
    is_rgb = _is_rgb_shape(shape)
    is_volume = (arr.ndim >= 3) and not is_rgb
    axes: Dict[str, Optional[int]] = {"z": None, "y": None, "x": None}
    if is_volume:
        axes["z"] = shape[0]
        axes["y"] = shape[1]
        axes["x"] = shape[2]
    elif arr.ndim == 2 or is_rgb:
        axes["y"] = shape[0]
        axes["x"] = shape[1]
    intensity_min = float(np.min(arr))
    intensity_max = float(np.max(arr))
    return AssetInfo(
        asset_id=asset.asset_id,
        display_name=asset.display_name,
        original_format=asset.original_format,
        metadata_text=asset.metadata_text,
        media_type=_detect_media_type(fmt, is_volume, is_rgb),
        file_size=file_size,
        ndim=int(arr.ndim),
        shape=list(shape),
        dtype=str(arr.dtype),
        intensity_min=intensity_min,
        intensity_max=intensity_max,
        is_rgb=is_rgb,
        is_volume=is_volume,
        axes=axes,
        extra={
            k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
            for k, v in (meta or {}).items()
            if k not in ("pixel_array",)
        },
    )


def _window(
    arr: np.ndarray,
    gamma: float = 1.0,
    contrast: float = 1.0,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Robust contrast windowing + optional gamma.

    Percentile-based windowing is what makes medical scans actually look like
    images — clipping the top/bottom 1% removes the long tail. Contrast and
    gamma are applied in normalized (0..1) space.
    """
    a = arr.astype(np.float32, copy=False)
    lo, hi = np.percentile(a, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    # Map [lo, hi] -> [0, 1]
    v = (a - lo) / (hi - lo)
    # Contrast = expand/compress around 0.5
    if contrast != 1.0:
        v = (v - 0.5) * contrast + 0.5
    v = np.clip(v, 0.0, 1.0)
    if gamma != 1.0 and gamma > 0:
        v = np.power(v, 1.0 / gamma)
    return (v * 255).astype(np.uint8)


def _to_rgb_uint8(arr: np.ndarray, gamma: float, contrast: float) -> np.ndarray:
    if _is_rgb_shape(arr.shape):
        # Apply gamma/contrast per channel.
        out = np.zeros_like(arr, dtype=np.uint8)
        for c in range(min(3, arr.shape[-1])):
            out[..., c] = _window(arr[..., c], gamma, contrast)
        if arr.shape[-1] == 4:
            out[..., 3] = arr[..., 3].astype(np.uint8)
        return out
    return _window(arr, gamma, contrast)


def _select_axis(arr: np.ndarray, axis: Axis) -> int:
    # We canonicalize axis-0=Z, axis-1=Y, axis-2=X.
    return {"z": 0, "y": 1, "x": 2}[axis]


def extract_slice(
    asset: Asset,
    axis: Axis = "z",
    index: int = 0,
    gamma: float = 1.0,
    contrast: float = 1.0,
) -> bytes:
    arr, _meta = _load_asset_array(asset)
    if arr.ndim == 2 or _is_rgb_shape(arr.shape):
        plane = arr
    else:
        ax = _select_axis(arr, axis)
        n = arr.shape[ax]
        index = max(0, min(n - 1, int(index)))
        plane = np.take(arr, index, axis=ax)
    img_arr = _to_rgb_uint8(plane, gamma, contrast)
    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def extract_mip(
    asset: Asset,
    axis: Axis = "z",
    gamma: float = 1.0,
    contrast: float = 1.0,
) -> bytes:
    arr, _meta = _load_asset_array(asset)
    if arr.ndim == 2 or _is_rgb_shape(arr.shape):
        plane = arr
    else:
        ax = _select_axis(arr, axis)
        plane = arr.max(axis=ax)
    img_arr = _to_rgb_uint8(plane, gamma, contrast)
    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def get_volume_bytes(
    asset: Asset, target_max_side: int = 96
) -> Tuple[bytes, list[int]]:
    """Return a downsampled Float32 volume as raw bytes + the new shape.

    Downsampled so the client doesn't have to ship the original full volume
    (some go past 100 MB). 96^3 ≈ 3.4 MB is plenty for an isosurface preview.
    """
    arr, _meta = _load_asset_array(asset)
    if arr.ndim < 3 or _is_rgb_shape(arr.shape):
        raise ValueError("not a volumetric asset")
    # Compute integer downsample factors per axis.
    factors = []
    for s in arr.shape[:3]:
        factors.append(max(1, int(np.ceil(s / target_max_side))))
    f0, f1, f2 = factors[0], factors[1], factors[2]
    # Crop to multiples then block-mean.
    cropped = arr[
        : (arr.shape[0] // f0) * f0,
        : (arr.shape[1] // f1) * f1,
        : (arr.shape[2] // f2) * f2,
    ]
    new_shape = (
        cropped.shape[0] // f0,
        cropped.shape[1] // f1,
        cropped.shape[2] // f2,
    )
    ds = (
        cropped.astype(np.float32)
        .reshape(new_shape[0], f0, new_shape[1], f1, new_shape[2], f2)
        .mean(axis=(1, 3, 5))
    )
    # Normalize to 0..1 (windowed) so the client doesn't have to know the
    # original intensity range.
    lo, hi = np.percentile(ds, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    ds = np.clip((ds - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return ds.tobytes(order="C"), [int(s) for s in new_shape]


def reset_cache() -> None:
    _volume_cache.clear()


__all__ = [
    "AssetInfo",
    "MediaType",
    "extract_mip",
    "extract_slice",
    "get_info",
    "get_volume_bytes",
    "reset_cache",
]
