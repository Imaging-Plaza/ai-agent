# utils/image_meta.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict

def _filesize_str(p: Path) -> str:
    try:
        b = p.stat().st_size
        for unit in ("B","KB","MB","GB","TB"):
            if b < 1024.0:
                return f"{b:.1f}{unit}"
            b /= 1024.0
    except Exception:
        pass
    return "?"

def summarize_image_metadata(path: str) -> str:
    """
    Build a short, human-readable metadata string for the selector prompt.
    Works for TIFF/PNG/JPG/WebP/BMP and NIfTI (.nii/.nii.gz). Fails gracefully.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".gz" and p.name.lower().endswith(".nii.gz"):
        ext = ".nii"
    parts = [f"filename={p.name}", f"ext={ext.lstrip('.')}", f"size={_filesize_str(p)}"]

    try:
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
            try:
                from PIL import Image, ImageSequence
            except Exception:
                return "; ".join(parts)
            with Image.open(str(p)) as im:
                n = getattr(im, "n_frames", 1)
                parts += [f"frames={n}", f"mode={im.mode}", f"size={getattr(im, 'size', None)}"]
                if ext in (".tif", ".tiff"):
                    # Optional richer TIFF info (if tifffile available)
                    try:
                        import tifffile as tiff
                        with tiff.TiffFile(str(p)) as tf:
                            page = tf.pages[0]
                            dtype = getattr(page.asarray(), "dtype", None)
                            comp = page.compression.name if getattr(page, "compression", None) else "unknown"
                            parts += [f"dtype={getattr(dtype, 'name', dtype)}", f"compression={comp}"]
                    except Exception:
                        pass
        elif ext == ".nii":
            try:
                import nibabel as nib
                img = nib.load(str(p))
                hdr = img.header
                shape = tuple(img.shape)
                zooms = tuple(float(z) for z in hdr.get_zooms()[:len(shape)])
                dtype = str(hdr.get_data_dtype())
                parts += [f"shape={shape}", f"zooms={zooms}", f"dtype={dtype}"]
            except Exception:
                pass
    except Exception:
        # fail quietly; we still include filename/ext/size
        pass

    return "; ".join(map(str, parts))

def detect_ext_token(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".gz" and p.name.lower().endswith(".nii.gz"):
        return "NII.GZ"
    return ext.lstrip(".").upper() if ext else None
