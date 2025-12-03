# utils/previews.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import tempfile
import logging
import time
import tifffile as tiff
from typing import List, Optional, Tuple
from ai_agent.utils.image_meta import summarize_image_metadata
from ai_agent.utils.image_io import load_any

log = logging.getLogger("pipeline")


def _norm_uint8(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.float32)
    v = v - np.nanmin(v)
    vmax = np.nanpercentile(v, 99.5) if np.isfinite(v).any() else 1.0
    vmax = vmax if vmax > 0 else (v.max() if v.max() > 0 else 1.0)
    v = np.clip(v / vmax, 0, 1)
    return (v * 255).astype(np.uint8)

def _is_rgb_like(shape: tuple[int, ...]) -> bool:
    """True for 2D color images shaped (H, W, 3/4)."""
    return len(shape) == 3 and shape[-1] in (3, 4) and shape[0] >= 16 and shape[1] >= 16

def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """Convert any numeric array to a uint8 image without changing shape."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating):
        if np.nanmax(a) <= 1.0:
            a = np.clip(a, 0.0, 1.0) * 255.0
        else:
            a = np.clip(a, 0.0, 255.0)
        return a.astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)

def mip_montage(vol3d: np.ndarray, out_png: str | Path) -> str:
    vol3d = _norm_uint8(vol3d)
    axial = vol3d.max(axis=2)
    cor = vol3d.max(axis=1)
    sag = vol3d.max(axis=0).T
    h1 = np.hstack([axial, cor])
    # pad to rectangle
    pad = np.zeros_like(axial)
    img = np.vstack([h1, np.hstack([sag, pad])])
    iio.imwrite(str(out_png), img)
    return str(out_png)

def slice_gif(vol: np.ndarray, out_gif: str | Path, axis: int = 2, step: int = 1, fps: int = 10) -> str:
    v = _norm_uint8(vol)
    idxs = list(range(0, v.shape[axis], step))
    frames = [np.take(v, i, axis=axis) for i in idxs]
    iio.imwrite(str(out_gif), frames, plugin="pillow", duration=int(1000 / fps), loop=0)
    return str(out_gif)

def stack_sweep_gif(vol3d: np.ndarray, out_gif: str | Path, fps: int = 12, max_frames: int = 64) -> str:
    v = _norm_uint8(vol3d)
    depth = v.shape[2]
    step = max(1, depth // max_frames)
    frames = [v[:, :, i] for i in range(0, depth, step)]
    iio.imwrite(str(out_gif), frames, plugin="pillow", duration=int(1000 / fps), loop=0)
    return str(out_gif)

def contact_sheet_slices(
    vol3d: np.ndarray,
    out_png: str | Path,
    max_slices: int = 36,
    grid_cols: int = 6,
) -> str:
    v = _norm_uint8(vol3d)
    depth = v.shape[2]
    step = max(1, depth // max_slices)
    frames = [v[:, :, i] for i in range(0, depth, step)]
    frames = frames[:max_slices]  # cap exactly

    # pad to full grid
    cols = grid_cols
    rows = int(np.ceil(len(frames) / cols))
    h, w = frames[0].shape
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = frame

    iio.imwrite(str(out_png), canvas)
    return str(out_png)

def _build_preview_for_vlm(image_paths: Optional[List[str]]) -> Tuple[Optional[str], Optional[str]]:
    if not image_paths:
        return None, None

    meta_text = None
    try:
        meta_text = summarize_image_metadata(image_paths)
    except Exception:
        log.exception("Image metadata summarization failed; continuing without metadata.")

    try:
        _cleanup_old_previews(hours=24)
    except Exception:
        pass

    for p in image_paths:
        try:
            data, meta = load_any(p)
            shp = getattr(meta, "shape", None) or meta.get("shape")
            if shp is None:
                shp = getattr(data, "shape", None)
            if shp is None:
                continue

            tmpdir = Path(tempfile.mkdtemp(prefix="preview_"))

            # Handle true color images (H, W, 3/4) safely
            arr = np.asarray(data)
            ext = Path(p).suffix.lower()

            # For PNG/JPEG/WebP, (H,W,3/4) is almost certainly color → render as-is
            if _is_rgb_like(arr.shape) and ext in {".png", ".jpg", ".jpeg", ".webp"}:
                out = tmpdir / "image.png"
                iio.imwrite(str(out), _to_uint8_image(arr))
                return str(out), meta_text

            # For TIFF, (H,W,3) can be either RGB or a 3-slice stack.
            # If tags say it's RGB, render as color; otherwise treat as stack (fall through).
            if _is_rgb_like(arr.shape) and ext in {".tif", ".tiff"}:
                try:
                    with tiff.TiffFile(p) as tf:
                        page = tf.pages[0]
                        spp = int(getattr(page, "samplesperpixel", 1))
                        photometric = str(getattr(page, "photometric", "")).upper()
                    if spp in (3, 4) and ("RGB" in photometric or "YCBCR" in photometric):
                        out = tmpdir / "image.png"
                        iio.imwrite(str(out), _to_uint8_image(arr))
                        return str(out), meta_text

                except Exception:
                    # If tags can't be read, prefer treating TIFF (H,W,3) as a stack
                    pass

            if len(shp) == 3:
                png_path = tmpdir / "slices_grid.png"
                gif_path = tmpdir / "sweep.gif"
                try:
                    contact_sheet_slices(arr, png_path, max_slices=36, grid_cols=6)
                except Exception:
                    try:
                        mip_montage(arr, png_path)
                    except Exception:
                        pass
                try:
                    stack_sweep_gif(arr, gif_path, fps=12, max_frames=64)
                except Exception:
                    pass
                if png_path.exists():
                    return str(png_path), meta_text
                if gif_path.exists():
                    return str(gif_path), meta_text

            if len(shp) == 4:
                vol = np.asarray(data).mean(axis=-1)
                out = tmpdir / "sweep.gif"
                step = max(1, vol.shape[2] // 64)
                slice_gif(vol, out, axis=2, step=step, fps=12)
                return str(out), meta_text

            if len(shp) == 2:
                out = tmpdir / "image.png"
                arr2 = np.asarray(data)
                if arr2.dtype != np.uint8:
                    arr2 = (np.clip(arr2, 0, 1) * 255).astype(np.uint8)
                iio.imwrite(str(out), arr2)
                return str(out), meta_text
        except Exception:
            continue

    return None, meta_text

def _cleanup_old_previews(hours: int = 24) -> None:
    """
    Delete preview_* folders older than `hours` from the system temp dir.
    Best-effort; ignore errors.
    """
    root = Path(tempfile.gettempdir())
    cutoff = time.time() - hours * 3600
    try:
        for p in root.glob("preview_*"):
            try:
                if p.is_dir() and p.stat().st_mtime < cutoff:
                    for sub in p.glob("**/*"):
                        try:
                            if sub.is_file():
                                sub.unlink()
                        except Exception:
                            pass
                    p.rmdir()
            except Exception:
                pass
    except Exception:
        logging.getLogger("api").exception("Preview cleanup failed")