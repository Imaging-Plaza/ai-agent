# utils/image_meta.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import threading
import os

import nibabel as nib
import pydicom
from PIL import Image
import tifffile as tiff

# ---------------------------------------------------------------------------
# In-process metadata cache (keyed by resolved-path + mtime + size)
# Avoids re-reading large files (e.g. TIFF stacks) on every retrieval call.
# ---------------------------------------------------------------------------
_META_CACHE_MAX = int(os.getenv("IMAGE_META_CACHE_MAX", "128"))
_meta_cache: dict[tuple, str] = {}  # key -> result string
_meta_cache_order: list[tuple] = []  # insertion-order for simple LRU eviction
_meta_cache_lock = threading.Lock()


def _meta_cache_key(p: Path) -> tuple:
    """Stable cache key: (resolved_path_str, mtime_ns, size_bytes)."""
    try:
        st = p.stat()
        return (str(p.resolve()), st.st_mtime_ns, st.st_size)
    except Exception:
        return (str(p), 0, 0)


def _meta_cache_get(key: tuple) -> Optional[str]:
    with _meta_cache_lock:
        return _meta_cache.get(key)


def _meta_cache_set(key: tuple, value: str) -> None:
    with _meta_cache_lock:
        if key in _meta_cache:
            return
        _meta_cache[key] = value
        _meta_cache_order.append(key)
        # Evict oldest entries when over capacity
        while len(_meta_cache_order) > _META_CACHE_MAX:
            oldest = _meta_cache_order.pop(0)
            _meta_cache.pop(oldest, None)

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
    """More robust DICOM detection"""
    try:
        # First try extension
        if p.suffix.lower() == ".dcm":
            return True

        # Then try DICM magic number
        with open(p, "rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True

        # Finally try loading with pydicom
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            return hasattr(ds, "SOPClassUID")
        except:
            pass

        return False
    except Exception:
        return False


def _is_dicom_path(p: Path) -> bool:
    if p.is_dir():
        return True  # heuristic: treat dirs as DICOM series candidates
    return _is_dicom_file(p)


# ---- summarizers -------------------------------------------------------------


def _summarize_nifti(p: Path) -> Optional[str]:
    try:
        img = nib.load(str(p))
        hdr = img.header
        shape = tuple(int(x) for x in img.shape)
        zooms = tuple(float(z) for z in hdr.get_zooms()[: len(shape)])
        dtype = str(hdr.get_data_dtype())
        return (
            f"NIfTI {len(shape)}D {shape} @ "
            + (("×".join(f"{z:.2f}" for z in zooms[:3]) + " mm") if zooms else "?")
            + f" dtype={dtype} filename={p.name} size={_filesize_str(p)}"
        )
    except Exception:
        # fall back to a minimal line
        return f"NIfTI ?D ? filename={p.name} size={_filesize_str(p)}"


def _summarize_dicom(path: Path) -> Optional[str]:
    """
    Summarize a DICOM series (dir) or single file without loading pixels.

    XA/fluoro handling:
      - Pixel spacing from Shared/Per-Frame Functional Groups if present.
      - Fall back to PixelSpacing and ImagerPixelSpacing.
      - Cine-style objects reported as 'frames'; add fps when possible.
      - Treat empty strings like missing values.
    """

    # ---------- helpers ----------
    def _nz(v):
        """None or empty/whitespace -> None; else strip strings."""
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    def _first(*vals):
        for v in vals:
            v = _nz(v)
            if v is not None:
                return v
        return None

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _code_meaning(seq):
        try:
            if seq and len(seq) > 0:
                item = seq[0]
                return _first(
                    getattr(item, "CodeMeaning", None), getattr(item, "CodeValue", None)
                )
        except Exception:
            pass
        return None

    def _px_from_fgs(ds):
        """(sy, sx, sz) from functional groups if present."""
        sy = sx = sz = None
        try:
            sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
            if sfg and len(sfg) > 0:
                pms = getattr(sfg[0], "PixelMeasuresSequence", None)
                if pms and len(pms) > 0:
                    p = pms[0]
                    ps = getattr(p, "PixelSpacing", None)
                    if isinstance(ps, (list, tuple)) and len(ps) == 2:
                        sy, sx = _safe_float(ps[0]), _safe_float(ps[1])
                    sz = _first(
                        _safe_float(getattr(p, "SpacingBetweenSlices", None)),
                        _safe_float(getattr(p, "SliceThickness", None)),
                    )
            if sy is None or sx is None:
                pfg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
                if pfg and len(pfg) > 0:
                    for it in pfg[:8]:
                        pms = getattr(it, "PixelMeasuresSequence", None)
                        if pms and len(pms) > 0:
                            p = pms[0]
                            ps = getattr(p, "PixelSpacing", None)
                            if isinstance(ps, (list, tuple)) and len(ps) == 2:
                                sy = _safe_float(ps[0]) if sy is None else sy
                                sx = _safe_float(ps[1]) if sx is None else sx
                            if sz is None:
                                sz = _first(
                                    _safe_float(
                                        getattr(p, "SpacingBetweenSlices", None)
                                    ),
                                    _safe_float(getattr(p, "SliceThickness", None)),
                                )
                            if sy is not None and sx is not None:
                                break
        except Exception:
            pass
        return sy, sx, sz

    def _anatomy_from_fgs(ds):
        try:
            sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
            if sfg and len(sfg) > 0:
                fas = getattr(sfg[0], "FrameAnatomySequence", None)
                if fas and len(fas) > 0:
                    return _code_meaning(
                        getattr(fas[0], "AnatomicRegionSequence", None)
                    )
            pfg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
            if pfg and len(pfg) > 0:
                fas = getattr(pfg[0], "FrameAnatomySequence", None)
                if fas and len(fas) > 0:
                    return _code_meaning(
                        getattr(fas[0], "AnatomicRegionSequence", None)
                    )
        except Exception:
            pass
        return None

    # ---------- gather a few headers ----------
    try:
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
            files = files[:256]
        else:
            files = [path]

        dsets = []
        for fp in files:
            try:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                if (
                    getattr(ds, "SOPClassUID", None) is None
                    and "_is_dicom_file" in globals()
                ):
                    if not _is_dicom_file(fp):
                        continue
                dsets.append(ds)
            except Exception:
                continue

        if not dsets:
            tag = "DIR" if path.is_dir() else "FILE"
            return f"DICOM ({tag}) filename={path.name}"

        ds0 = dsets[0]

        modality = _first(getattr(ds0, "Modality", None), "?").upper()
        rows = getattr(ds0, "Rows", None)
        cols = getattr(ds0, "Columns", None)
        size_txt = f"{cols}x{rows}" if rows and cols else "?"

        # frames vs slices
        try:
            n_frames = int(_first(getattr(ds0, "NumberOfFrames", None), 0) or 0)
        except Exception:
            n_frames = 0
        cine_like = (n_frames > 1) or modality in {"XA", "XRF", "US", "RF"}
        n_items = n_frames if n_frames > 0 else len(dsets)
        count_label = "frames" if cine_like else "slices"

        # spacing (sy, sx, sz)
        sy, sx, sz = _px_from_fgs(ds0)
        if sy is None or sx is None:
            px = getattr(ds0, "PixelSpacing", None)
            if isinstance(px, (list, tuple)) and len(px) == 2:
                sy, sx = _safe_float(px[0]), _safe_float(px[1])
        if sy is None or sx is None:
            ipx = getattr(ds0, "ImagerPixelSpacing", None)  # common in XA
            if isinstance(ipx, (list, tuple)) and len(ipx) == 2:
                sy = _safe_float(ipx[0]) if sy is None else sy
                sx = _safe_float(ipx[1]) if sx is None else sx

        # Z spacing (rarely meaningful for XA cine)
        if not cine_like and sz is None and len(dsets) > 1:
            try:
                zvals = []
                for ds in dsets[:64]:
                    ipp = getattr(ds, "ImagePositionPatient", None)
                    if isinstance(ipp, (list, tuple)) and len(ipp) == 3:
                        z = _safe_float(ipp[2])
                        if z is not None:
                            zvals.append(z)
                if len(zvals) >= 2:
                    zvals.sort()
                    diffs = [
                        abs(zvals[i + 1] - zvals[i]) for i in range(len(zvals) - 1)
                    ]
                    if diffs:
                        sz = sum(diffs) / len(diffs)
            except Exception:
                pass
        if not cine_like and sz is None:
            sz = _first(
                getattr(ds0, "SpacingBetweenSlices", None),
                getattr(ds0, "SliceThickness", None),
            )
            sz = _safe_float(sz)

        # body / series (with better fallbacks)
        body = _first(
            getattr(ds0, "BodyPartExamined", None),
            _code_meaning(getattr(ds0, "AnatomicRegionSequence", None)),
            _anatomy_from_fgs(ds0),
            getattr(ds0, "RequestedProcedureDescription", None),
            getattr(ds0, "StudyDescription", None),
            "?",
        )
        series = _first(
            getattr(ds0, "SeriesDescription", None),
            getattr(ds0, "ProtocolName", None),
            getattr(ds0, "PerformedProcedureStepDescription", None),
            getattr(ds0, "StudyDescription", None),
            "?",
        )

        # cine timing
        fps = None
        cine_rate = _safe_float(getattr(ds0, "CineRate", None))
        if cine_rate and cine_rate > 0:
            fps = cine_rate
        else:
            ft = _safe_float(getattr(ds0, "FrameTime", None))  # ms
            if ft and ft > 0:
                fps = 1000.0 / ft
            else:
                # FrameTimeVector fallback: average of first few
                try:
                    ftv = getattr(ds0, "FrameTimeVector", None)
                    if ftv:
                        vals = [_safe_float(v) for v in list(ftv)[:16]]
                        vals = [v for v in vals if v and v > 0]
                        if vals:
                            fps = 1000.0 / (sum(vals) / len(vals))
                except Exception:
                    pass

        # spacing text
        if sy is not None and sx is not None and (sz is not None and not cine_like):
            sp_txt = f"{sy:.2f}×{sx:.2f}×{sz:.2f} mm"
        elif sy is not None and sx is not None:
            sp_txt = f"{sy:.2f}×{sx:.2f} mm"
        else:
            sp_txt = "N/A"

        extras = []
        if fps:
            extras.append(f"fps≈{fps:.1f}")
        extras_txt = (" " + " ".join(extras)) if extras else ""

        scope = "DIR" if path.is_dir() else "FILE"
        return (
            f"DICOM {modality} {scope} "
            f"{count_label}≈{n_items} size={size_txt} @ {sp_txt}{extras_txt} "
            f"body={body} series='{series}' name={path.name}"
        )
    except Exception:
        scope = "DIR" if path.is_dir() else "FILE"
        return f"DICOM {scope} name={path.name}"


def _summarize_image(p: Path) -> Optional[str]:
    """
    Summarize PNG/JPEG/TIFF (including TIFF stacks) via Pillow.
    """
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
                    with tiff.TiffFile(str(p)) as tf:
                        page = tf.pages[0]
                        arr = page.asarray()
                        dtype_txt = f" dtype={getattr(arr, 'dtype', '')}"
                        comp = getattr(page, "compression", None)
                        if comp:
                            comp_txt = (
                                f" compression={getattr(comp, 'name', str(comp))}"
                            )
                except Exception:
                    pass
            return f"{fmt} {'stack' if n>1 else 'image'} frames={n} size={size} mode={mode}{dtype_txt}{comp_txt} filename={p.name} size={_filesize_str(p)}"
    except Exception:
        return f"{p.suffix.upper().lstrip('.')} ? filename={p.name} size={_filesize_str(p)}"


# ---- public API --------------------------------------------------------------


def summarize_image_metadata(
    paths: Optional[List[str]] | Optional[str],
) -> Optional[str]:
    """
    Build a short, human-readable summary for one path or a list of paths.
    - DICOM dir/file: uses pydicom (tags only) and estimates slices & spacing.
    - NIfTI: uses nibabel to get shape/zooms/dtype.
    - Images (PNG/JPEG/TIFF...): uses Pillow, recognizes TIFF stacks.
    This function is robust: failures result in minimal per-item summaries.

    Results are cached in-process by (path, mtime_ns, size) so repeated calls
    within or across requests do not re-read the same file.
    """
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]

    parts: List[str] = []
    for s in paths:
        try:
            p = Path(s)
            cache_key = _meta_cache_key(p)
            cached = _meta_cache_get(cache_key)
            if cached is not None:
                parts.append(cached)
                continue

            if p.is_dir() or _is_dicom_path(p):
                result = _summarize_dicom(p) or f"DICOM name={p.name}"
            elif _is_nifti_path(p):
                result = _summarize_nifti(p) or f"NIfTI name={p.name}"
            else:
                result = (
                    _summarize_image(p)
                    or f"{p.suffix.upper().lstrip('.')} name={p.name}"
                )
            _meta_cache_set(cache_key, result)
            parts.append(result)
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
