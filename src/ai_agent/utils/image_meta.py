# utils/image_meta.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import os

# ---- small helpers -----------------------------------------------------------

def _filesize_str(p: Path) -> str:
    try:
        b = p.stat().st_size
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024.0:
                return f"{b:.1f}{unit}"
            b /= 1024.0
    except Exception:
        pass
    return "?"

def _is_nifti_path(p: Path) -> bool:
    s = p.name.lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")

def _is_dicom_file(p: Path) -> bool:
    # quick checks: extension or DICM magic
    s = p.suffix.lower()
    if s == ".dcm":
        return True
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False

def _is_dicom_path(p: Path) -> bool:
    if p.is_dir():
        return True  # heuristic: treat dirs as DICOM series candidates
    return _is_dicom_file(p)

# ---- summarizers -------------------------------------------------------------

def _summarize_nifti(p: Path) -> Optional[str]:
    try:
        import nibabel as nib
        img = nib.load(str(p))
        hdr = img.header
        shape = tuple(int(x) for x in img.shape)
        zooms = tuple(float(z) for z in hdr.get_zooms()[:len(shape)])
        dtype = str(hdr.get_data_dtype())
        return f"NIfTI {len(shape)}D {shape} @ " + \
               (("×".join(f"{z:.2f}" for z in zooms[:3]) + " mm") if zooms else "?") + \
               f" dtype={dtype} filename={p.name} size={_filesize_str(p)}"
    except Exception:
        # fall back to a minimal line
        return f"NIfTI ?D ? filename={p.name} size={_filesize_str(p)}"

def _summarize_dicom(path: Path) -> Optional[str]:
    """
    Summarize a DICOM series (dir) or single file using pydicom without loading pixels.
    If a directory is given, scan a subset of files for speed.
    """
    try:
        import pydicom  # type: ignore
    except Exception:
        # Can't parse tags; return minimal hint
        tag = "DIR" if path.is_dir() else "FILE"
        return f"DICOM ({tag}) filename={path.name} size={_filesize_str(path) if path.is_file() else '?'}"

    try:
        files: List[Path]
        if path.is_dir():
            # pick up to ~64 files likely belonging to one series
            files = [p for p in path.rglob("*") if p.is_file()]
            files = files[:256]
        else:
            files = [path]

        dsets = []
        for fp in files:
            try:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                # skip non-DICOMs quietly
                if getattr(ds, "SOPClassUID", None) is None and not _is_dicom_file(fp):
                    continue
                dsets.append(ds)
            except Exception:
                continue

        if not dsets:
            tag = "DIR" if path.is_dir() else "FILE"
            return f"DICOM ({tag}) filename={path.name}"

        ds0 = dsets[0]
        modality = getattr(ds0, "Modality", "?")
        body = getattr(ds0, "BodyPartExamined", None) or getattr(ds0, "StudyDescription", None) or "?"
        series = getattr(ds0, "SeriesDescription", None) or "?"
        # Best-effort in-plane spacing
        try:
            pxsp = getattr(ds0, "PixelSpacing", [None, None])
            sy, sx = (float(pxsp[0]) if pxsp[0] else None, float(pxsp[1]) if pxsp[1] else None)
        except Exception:
            sy = sx = None

        # Estimate slice count and spacing using ImagePositionPatient
        n_slices = len(dsets)
        sz = None
        try:
            # find two slices with IPP
            zvals = []
            for ds in dsets[:64]:
                ipp = getattr(ds, "ImagePositionPatient", None)
                if isinstance(ipp, (list, tuple)) and len(ipp) == 3:
                    zvals.append(float(ipp[2]))
            if len(zvals) >= 2:
                zvals.sort()
                diffs = [abs(zvals[i+1] - zvals[i]) for i in range(len(zvals)-1)]
                if diffs:
                    sz = sum(diffs) / len(diffs)
            if sz is None:
                sz = float(getattr(ds0, "SpacingBetweenSlices",
                                   getattr(ds0, "SliceThickness", None)))
        except Exception:
            pass

        sp_txt = "×".join([
            f"{sy:.2f}" if isinstance(sy, float) else "?",
            f"{sx:.2f}" if isinstance(sx, float) else "?",
            f"{sz:.2f}" if isinstance(sz, float) else "?"
        ]) + " mm"

        scope = "DIR" if path.is_dir() else "FILE"
        return f"DICOM {modality} {scope} slices≈{n_slices} @ {sp_txt} body={body} series='{series}' name={path.name}"
    except Exception:
        scope = "DIR" if path.is_dir() else "FILE"
        return f"DICOM {scope} name={path.name}"

def _summarize_image(p: Path) -> Optional[str]:
    """
    Summarize PNG/JPEG/TIFF (including TIFF stacks) via Pillow.
    """
    try:
        from PIL import Image
    except Exception:
        return f"{p.suffix.upper().lstrip('.')} ?D ? filename={p.name} size={_filesize_str(p)}"

    try:
        with Image.open(str(p)) as im:
            n = getattr(im, "n_frames", 1)
            size = getattr(im, "size", None)
            mode = im.mode
            fmt = p.suffix.upper().lstrip(".")
            # Try to get dtype/compression for TIFF if tifffile is available
            dtype_txt = comp_txt = ""
            if p.suffix.lower() in (".tif", ".tiff"):
                try:
                    import tifffile as tiff  # type: ignore
                    with tiff.TiffFile(str(p)) as tf:
                        page = tf.pages[0]
                        arr = page.asarray()
                        dtype_txt = f" dtype={getattr(arr, 'dtype', '')}"
                        comp = getattr(page, "compression", None)
                        if comp:
                            comp_txt = f" compression={getattr(comp, 'name', str(comp))}"
                except Exception:
                    pass
            return f"{fmt} {'stack' if n>1 else 'image'} frames={n} size={size} mode={mode}{dtype_txt}{comp_txt} filename={p.name} size={_filesize_str(p)}"
    except Exception:
        return f"{p.suffix.upper().lstrip('.')} ? filename={p.name} size={_filesize_str(p)}"

# ---- public API --------------------------------------------------------------

def summarize_image_metadata(paths: Optional[List[str]] | Optional[str]) -> Optional[str]:
    """
    Build a short, human-readable summary for one path or a list of paths.
    - DICOM dir/file: uses pydicom (tags only) and estimates slices & spacing.
    - NIfTI: uses nibabel to get shape/zooms/dtype.
    - Images (PNG/JPEG/TIFF...): uses Pillow, recognizes TIFF stacks.
    This function is robust: failures result in minimal per-item summaries.
    """
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]

    parts: List[str] = []
    for s in paths:
        try:
            p = Path(s)
            if p.is_dir() or _is_dicom_path(p):
                parts.append(_summarize_dicom(p) or f"DICOM name={p.name}")
            elif _is_nifti_path(p):
                parts.append(_summarize_nifti(p) or f"NIfTI name={p.name}")
            else:
                parts.append(_summarize_image(p) or f"{p.suffix.upper().lstrip('.')} name={p.name}")
        except Exception as e:
            parts.append(f"Unreadable '{s}': {e.__class__.__name__}")
    return " | ".join(parts)

def detect_ext_token(paths: Optional[List[str]] | Optional[str]) -> Optional[str]:
    """
    Return a space-separated string of canonical format tokens among the inputs:
    e.g., "DICOM NIfTI TIFF". Useful to bias retrieval.
    """
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]

    tokens = set()
    for s in paths:
        p = Path(s)
        if p.is_dir() or _is_dicom_path(p) or p.suffix.lower() == ".dcm":
            tokens.add("DICOM")
            continue
        if _is_nifti_path(p):
            tokens.add("NIfTI")
            continue
        ext = p.suffix.lower()
        if ext in (".tif", ".tiff"):
            tokens.add("TIFF")
        elif ext in (".png",):
            tokens.add("PNG")
        elif ext in (".jpg", ".jpeg"):
            tokens.add("JPEG")
        elif ext in (".bmp",):
            tokens.add("BMP")
        elif ext in (".webp",):
            tokens.add("WEBP")
    return " ".join(sorted(tokens)) if tokens else None