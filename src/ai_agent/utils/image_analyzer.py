from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import nibabel as nib

log = logging.getLogger("perception.vlm")


def _pil_to_png_bytes(im) -> bytes:
    # Ensure 8-bit single-channel or RGB
    if im.mode not in ("L", "RGB"):
        im = im.convert("L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _numpy_to_png_bytes(arr) -> bytes:
    # robust normalize [p2, p98] -> [0,255]
    a = arr.astype("float32")
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max() if a.max() > a.min() else a.min() + 1.0)
    a = (a - lo) / (hi - lo)
    a = (a.clip(0, 1) * 255).astype("uint8")
    im = Image.fromarray(a)
    return _pil_to_png_bytes(im)


def _tif_to_png_bytes(path: str) -> bytes:
    with Image.open(path) as im:
        # If multi-page/3D, pick middle frame
        n = getattr(im, "n_frames", 1)
        if n > 1:
            mid = n // 2
            im.seek(mid)
        return _pil_to_png_bytes(im)


def _nii_to_png_bytes(path: str) -> bytes:
    img = nib.load(path)
    data = img.get_fdata()
    # pick middle axial slice
    if data.ndim == 3:
        z = data.shape[2] // 2
        slice2d = data[:, :, z]
    elif data.ndim == 4:
        z = data.shape[2] // 2
        t0 = 0
        slice2d = data[:, :, z, t0]
    else:
        raise RuntimeError(f"Unsupported NIfTI ndim={data.ndim}")
    return _numpy_to_png_bytes(slice2d)


def _to_supported_png_dataurl(image_path: str) -> Optional[str]:
    """
    Convert various formats to a PNG data URL suitable for OpenAI VLM:
    supports tif/tiff, png, jpg/jpeg, webp, and nifti (nii/nii.gz).
    Returns data URL string or None if conversion fails.
    """
    p = Path(image_path)
    ext = p.suffix.lower()
    if ext == ".gz" and p.name.lower().endswith(".nii.gz"):
        ext = ".nii"

    try:
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            # Re-encode to PNG to be safe
            with Image.open(image_path) as im:
                png = _pil_to_png_bytes(im)
        elif ext in (".tif", ".tiff"):
            png = _tif_to_png_bytes(image_path)
        elif ext in (".nii",):
            png = _nii_to_png_bytes(image_path)
        else:
            # try generic PIL open and encode to PNG
            with Image.open(image_path) as im:
                png = _pil_to_png_bytes(im)
    except Exception as e:
        log.warning("Failed to convert image for VLM (%s): %s", image_path, e)
        return None

    b64 = base64.b64encode(png).decode("utf-8")
    return f"data:image/png;base64,{b64}"
