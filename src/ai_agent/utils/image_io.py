# utils/image_io.py
from __future__ import annotations
from pathlib import Path
import shutil
import time
import tempfile
import zipfile
import numpy as np
import imageio.v3 as iio
import pydicom
import nibabel as nib
from typing import Tuple, Dict, Any
from pydicom.pixels import apply_modality_lut, apply_voi_lut


def is_dicom_path(path: str | Path) -> bool:
    """Improved DICOM detection"""
    p = Path(path)
    if p.is_dir():
        # Check if directory contains any .dcm files
        return any(f.suffix.lower() == ".dcm" for f in p.rglob("*"))

    # For single files, do proper DICOM validation
    try:
        pydicom.dcmread(str(p), stop_before_pixels=True)
        return True
    except Exception:
        return False


def _safe_rmtree(p: Path) -> None:
    """Remove temp directory if it was created by us (safety guard)."""
    try:
        p = Path(p)
        troot = Path(tempfile.gettempdir())
        if (
            p.is_dir()
            and p.parent == troot
            and (p.name.startswith("dicom_zip_") or p.name.startswith("preview_"))
        ):
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _cleanup_old_dicom_zips(hours: int = 6) -> None:
    """Cleanup stale dicom_zip_* temp folders older than `hours`. Best-effort."""
    troot = Path(tempfile.gettempdir())
    cutoff = time.time() - hours * 3600
    try:
        for d in troot.glob("dicom_zip_*"):
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


def maybe_unzip(path: str | Path) -> Path:
    """Safely extract zip file to temp directory, with better error handling."""
    p = Path(path)
    if p.is_dir() or p.suffix.lower() != ".zip":
        return p

    try:
        _cleanup_old_dicom_zips(hours=6)

        tmp = Path(tempfile.mkdtemp(prefix="dicom_zip_"))
        with zipfile.ZipFile(p) as z:
            # Check if zip contains DICOM files
            has_dicom = any(name.lower().endswith(".dcm") for name in z.namelist())
            if not has_dicom:
                raise ValueError("ZIP file contains no DICOM files")

            # Extract with path sanitization
            for item in z.namelist():
                if ".." not in item:  # Basic path traversal protection
                    z.extract(item, tmp)
        return tmp
    except Exception as e:
        raise ValueError(f"Failed to process ZIP file: {str(e)}")


def load_nifti(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    hdr = img.header
    zooms = tuple(float(z) for z in hdr.get_zooms())
    return data, {
        "format": "NIfTI",
        "shape": data.shape,
        "zooms": zooms,
        "datatype": str(hdr.get_data_dtype()),
    }


def load_dicom_series(dir_or_file: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    root = Path(dir_or_file)
    if root.is_file():
        # Could be a single multi-frame file; keep it as-is
        files = [root]
    else:
        files = sorted([p for p in root.rglob("*") if p.is_file()])

    dsets = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False, force=True)
            if hasattr(ds, "PixelData"):
                dsets.append(ds)
        except Exception:
            continue
    if not dsets:
        raise ValueError("No DICOM slices with pixel data found.")

    # If the first file is multi-frame, load frames from it
    first = dsets[0]

    def _prep_pixels(ds):
        arr = ds.pixel_array  # may be (frames, rows, cols) or (rows, cols)
        # Apply LUTs/windowing for proper display range
        try:
            arr = apply_modality_lut(arr, ds)
        except Exception:
            pass
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        arr = arr.astype(np.float32)

        # Handle MONOCHROME1 (invert)
        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            # invert per frame
            arr = arr.max() - arr

        return arr

    if getattr(first, "NumberOfFrames", None):
        arr = _prep_pixels(first)
        # ensure (H, W, Z)
        if arr.ndim == 3:  # (frames, rows, cols)
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:  # (rows, cols)
            arr = arr[..., None]
        vol = arr
        dsets = [first]  # meta from first
    else:
        # classic per-slice series
        def sort_key(ds):
            inst = getattr(ds, "InstanceNumber", 0)
            try:
                inst = int(inst)
            except Exception:
                inst = 0
            ipp = getattr(ds, "ImagePositionPatient", None)
            z = (
                float(ipp[2])
                if (isinstance(ipp, (list, tuple)) and len(ipp) == 3)
                else 0.0
            )
            return (inst, z)

        dsets.sort(key=sort_key)
        frames = [_prep_pixels(ds) for ds in dsets]

        # each frame is (H,W); make (H,W,Z)
        vol = np.stack([f if f.ndim == 2 else f[0] for f in frames], axis=-1).astype(
            np.float32
        )

    # Normalize to [0,1] for consistent downstream PNG saving
    vmin = np.nanmin(vol)
    vmax = np.nanmax(vol)
    if np.isfinite(vmax) and vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)

    # Spacing
    pxsp = getattr(dsets[0], "PixelSpacing", None)
    if not (isinstance(pxsp, (list, tuple)) and len(pxsp) == 2):
        # XA often uses ImagerPixelSpacing instead
        pxsp = getattr(dsets[0], "ImagerPixelSpacing", None)
    sy, sx = (float(pxsp[0]), float(pxsp[1])) if pxsp else (1.0, 1.0)

    # Slice spacing
    if vol.shape[-1] > 1:
        try:
            # use IPP difference if available
            z0 = float(getattr(dsets[0], "ImagePositionPatient", [0, 0, 0])[2])
            z1 = float(
                getattr(
                    dsets[min(1, len(dsets) - 1)], "ImagePositionPatient", [0, 0, 0]
                )[2]
            )
            sz = (
                abs(z1 - z0)
                if z1 != z0
                else float(getattr(dsets[0], "SliceThickness", 1.0))
            )
        except Exception:
            sz = float(getattr(dsets[0], "SliceThickness", 1.0))
    else:
        sz = float(getattr(dsets[0], "SliceThickness", 1.0))

    meta = {
        "format": "DICOM",
        "shape": vol.shape,  # (H, W, Z)
        "spacing": (sy, sx, sz),  # mm
        "Modality": getattr(dsets[0], "Modality", None),
        "BodyPartExamined": getattr(dsets[0], "BodyPartExamined", None),
        "SeriesDescription": getattr(dsets[0], "SeriesDescription", None),
        "SeriesInstanceUID": getattr(dsets[0], "SeriesInstanceUID", None),
        "StudyDescription": getattr(dsets[0], "StudyDescription", None),
        "PatientSex": getattr(dsets[0], "PatientSex", None),
        "PatientAge": getattr(dsets[0], "PatientAge", None),
        "PhotometricInterpretation": getattr(
            dsets[0], "PhotometricInterpretation", None
        ),
        "NumberOfFrames": getattr(dsets[0], "NumberOfFrames", None),
        "BitsStored": getattr(dsets[0], "BitsStored", None),
    }
    return vol.astype(np.float32), meta


def load_any(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load any supported image format with better error handling and temp cleanup.

    If the input is a ZIP containing DICOMs, we extract to a temp dir and ensure
    it is deleted after loading.
    """
    temp_dir: Path | None = None
    try:
        p = maybe_unzip(path)
        # Track whether we created a temp extraction dir
        pt = Path(p)
        if (
            pt.is_dir()
            and pt.parent == Path(tempfile.gettempdir())
            and pt.name.startswith("dicom_zip_")
        ):
            temp_dir = pt

        # Handle DICOM
        if Path(p).is_dir() or is_dicom_path(p):
            data, meta = load_dicom_series(p)
            return data, meta

        # Handle NIfTI
        s = str(p).lower()
        if s.endswith(".nii") or s.endswith(".nii.gz"):
            return load_nifti(p)

        # Handle regular images
        arr = iio.imread(str(p))
        meta = {"format": Path(p).suffix.upper().lstrip("."), "shape": arr.shape}
        return arr.astype(np.float32), meta
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {str(e)}")
    finally:
        if temp_dir is not None:
            _safe_rmtree(temp_dir)
