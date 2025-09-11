# utils/image_io.py
from __future__ import annotations
import os, zipfile, tempfile
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import imageio.v3 as iio
import nibabel as nib
import pydicom

def is_dicom_path(path: str | Path) -> bool:
    p = Path(path)
    if p.is_dir():
        return True
    # quick magic check
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False

def maybe_unzip(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir() or p.suffix.lower() != ".zip":
        return p
    tmp = Path(tempfile.mkdtemp(prefix="dicom_zip_"))
    with zipfile.ZipFile(p) as z:
        z.extractall(tmp)
    return tmp

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
        root = root.parent
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

    # sort by InstanceNumber then by ImagePositionPatient z
    def sort_key(ds):
        inst = getattr(ds, "InstanceNumber", 0)
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if (isinstance(ipp, (list, tuple)) and len(ipp) == 3) else 0.0
        try:
            inst = int(inst)
        except Exception:
            inst = 0
        return (inst, z)

    dsets.sort(key=sort_key)
    vol = np.stack([ds.pixel_array for ds in dsets], axis=-1).astype(np.float32)  # (H,W,Z)

    # spacing
    pxsp = getattr(dsets[0], "PixelSpacing", [1.0, 1.0])
    sy, sx = [float(x) for x in pxsp] if isinstance(pxsp, (list, tuple)) and len(pxsp) == 2 else (1.0, 1.0)
    # slice spacing
    if hasattr(dsets[0], "SpacingBetweenSlices"):
        sz = float(dsets[0].SpacingBetweenSlices)
    else:
        try:
            if len(dsets) > 1 and hasattr(dsets[0], "ImagePositionPatient") and hasattr(dsets[1], "ImagePositionPatient"):
                z0 = float(dsets[0].ImagePositionPatient[2])
                z1 = float(dsets[1].ImagePositionPatient[2])
                sz = abs(z1 - z0)
            else:
                sz = float(getattr(dsets[0], "SliceThickness", 1.0))
        except Exception:
            sz = float(getattr(dsets[0], "SliceThickness", 1.0))

    meta = {
        "format": "DICOM",
        "shape": vol.shape,
        "spacing": (sy, sx, sz),  # row, col, slice (mm)
        "Modality": getattr(dsets[0], "Modality", None),
        "BodyPartExamined": getattr(dsets[0], "BodyPartExamined", None),
        "SeriesDescription": getattr(dsets[0], "SeriesDescription", None),
        "SeriesInstanceUID": getattr(dsets[0], "SeriesInstanceUID", None),
        "StudyDescription": getattr(dsets[0], "StudyDescription", None),
        "PatientSex": getattr(dsets[0], "PatientSex", None),
        "PatientAge": getattr(dsets[0], "PatientAge", None),
    }
    return vol, meta

def load_any(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = maybe_unzip(path)
    if Path(p).is_dir() or is_dicom_path(p):
        return load_dicom_series(p)
    s = str(p).lower()
    if s.endswith(".nii") or s.endswith(".nii.gz"):
        return load_nifti(p)
    # 2D image or TIFF stack -> imageio
    arr = iio.imread(str(p))
    meta = {"format": Path(p).suffix.upper().lstrip("."), "shape": arr.shape}
    return arr.astype(np.float32), meta
