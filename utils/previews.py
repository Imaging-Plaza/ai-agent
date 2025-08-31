# utils/previews.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v3 as iio

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
