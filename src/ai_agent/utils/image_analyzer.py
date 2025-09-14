from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
import nibabel as nib
from openai import OpenAI

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


def analyze_image_with_text(image_path: str, user_task: str) -> Dict[str, Any]:
    """
    Uses a VLM (OpenAI) to extract cues from the image *and* the user's text.
    Returns a dict like:
      {"modality":"CT","dims":"3D","anatomy":"lung","suspected_task":"segmentation","keywords":[...]}

    If no OPENAI_API_KEY or conversion fails, returns {} so pipeline can proceed.
    """
    if not os.getenv("OPENAI_API_KEY"):
        log.info("OPENAI_API_KEY not set; skipping VLM analysis.")
        return {}

    # Convert to PNG data URL supported by the API
    data_url = _to_supported_png_dataurl(image_path)
    if not data_url:
        # Don’t block the pipeline if we cannot convert (e.g., missing libs)
        log.warning("VLM: unsupported image format; skipping image cues.")
        return {}

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = os.getenv("OPENAI_VLM_MODEL", "gpt-4o-mini")
    log.info("VLM model=%s", model_name)

    system = (
        "You are a medical+imaging assistant. "
        "Given an image and a short task description, infer concise retrieval cues. "
        "Respond as a compact JSON object with keys: "
        "modality (CT/MRI/XR/US/natural/other), "
        "dims (2D/3D/unknown), anatomy (one word or ''), suspected_task "
        "(segmentation/deblurring/registration/classification/other), and keywords (3-8 short nouns)."
    )
    user = f"Task: {user_task}\nReturn JSON only."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    _save_prompt(kind="vlm_cues", model=model_name, system=system, user_text=user, data_url=data_url)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        cues = json.loads(text)
        out = {
            "modality": (cues.get("modality") or "").upper(),
            "dims": (cues.get("dims") or "").upper(),
            "anatomy": (cues.get("anatomy") or "").lower(),
            "suspected_task": (cues.get("suspected_task") or "").lower(),
            "keywords": cues.get("keywords") or [],
        }
        return out
    except Exception as e:
        log.warning("VLM analysis failed: %s", e)
        return {}


def _save_prompt(kind: str, model: str, system: str, user_text: str,
                 data_url: Optional[str]) -> Optional[str]:
    if str(os.getenv("LOG_PROMPTS", "")).lower() not in ("1", "true", "yes", "on"):
        return None
    Path("logs").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = Path("logs") / f"{kind}_{ts}.txt"
    png_path = None
    if data_url and data_url.startswith("data:image/png;base64,"):
        try:
            b64 = data_url.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            png_path = Path("logs") / f"{kind}_{ts}.png"
            png_path.write_bytes(img_bytes)
        except Exception:
            png_path = None
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"MODEL: {model}\n\n")
        f.write("--- SYSTEM ---\n")
        f.write(system.strip() + "\n\n")
        f.write("--- USER (text) ---\n")
        f.write(user_text.strip() + "\n\n")
        if png_path:
            f.write(f"--- IMAGE ---\nSaved PNG: {png_path}\n")
        elif data_url:
            f.write(f"--- IMAGE ---\n(data-url length: {len(data_url)})\n")
    return str(txt_path)
