# utils/previews.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import tempfile
import logging
import time
from typing import List, Optional, Tuple
from utils.image_meta import summarize_image_metadata
from utils.image_io import load_any

log = logging.getLogger("pipeline")


def _norm_uint8(a: np.ndarray) -> np.ndarray:
    v = a.astype(np.float32)
    v = v - np.nanmin(v)
    vmax = np.nanpercentile(v, 99.5) if np.isfinite(v).any() else 1.0
    vmax = vmax if vmax > 0 else (v.max() if v.max() > 0 else 1.0)
    v = np.clip(v / vmax, 0, 1)
    return (v * 255).astype(np.uint8)

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

    for p in image_paths:
        try:
            data, meta = load_any(p)
            shp = getattr(meta, "shape", None) or meta.get("shape")
            if shp is None:
                shp = getattr(data, "shape", None)
            if shp is None:
                continue

            tmpdir = Path(tempfile.mkdtemp(prefix="preview_"))

            if len(shp) == 3:
                png_path = tmpdir / "slices_grid.png"
                gif_path = tmpdir / "sweep.gif"
                try:
                    contact_sheet_slices(data, png_path, max_slices=36, grid_cols=6)
                except Exception:
                    try:
                        mip_montage(data, png_path)
                    except Exception:
                        pass
                try:
                    stack_sweep_gif(data, gif_path, fps=12, max_frames=64)
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
                arr = data
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                iio.imwrite(str(out), arr)
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