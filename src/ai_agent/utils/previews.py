# utils/previews.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import tempfile
import logging
import time
import tifffile as tiff
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
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

def create_orthogonal_views(
    vol3d: np.ndarray,
    out_png: str | Path,
    annotations: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a comprehensive 3-view (axial, coronal, sagittal) visualization.
    Each view shows both a middle slice and a MIP projection.
    
    Args:
        vol3d: 3D volume array
        out_png: Output path for PNG
        annotations: Optional metadata dict to overlay (format, modality, dims, etc.)
    """
    v = _norm_uint8(vol3d)
    h, w, d = v.shape
    
    # Middle slices
    axial_slice = v[:, :, d // 2]
    coronal_slice = v[:, w // 2, :]
    sagittal_slice = v[h // 2, :, :].T
    
    # MIP projections
    axial_mip = v.max(axis=2)
    coronal_mip = v.max(axis=1)
    sagittal_mip = v.max(axis=0).T
    
    # Ensure all views have similar aspect ratios by padding
    def pad_to_square(img: np.ndarray, target_size: int) -> np.ndarray:
        h, w = img.shape
        if h == w:
            return img
        pad_h = (target_size - h) // 2 if h < target_size else 0
        pad_w = (target_size - w) // 2 if w < target_size else 0
        return np.pad(img, ((pad_h, target_size - h - pad_h), (pad_w, target_size - w - pad_w)), mode='constant')
    
    max_dim = max(axial_slice.shape[0], axial_slice.shape[1],
                  coronal_slice.shape[0], coronal_slice.shape[1],
                  sagittal_slice.shape[0], sagittal_slice.shape[1])
    
    # Create 2x3 grid: MIPs on top row, slices on bottom row
    top_row = np.hstack([
        pad_to_square(axial_mip, max_dim),
        pad_to_square(coronal_mip, max_dim),
        pad_to_square(sagittal_mip, max_dim)
    ])
    
    bottom_row = np.hstack([
        pad_to_square(axial_slice, max_dim),
        pad_to_square(coronal_slice, max_dim),
        pad_to_square(sagittal_slice, max_dim)
    ])
    
    composite = np.vstack([top_row, bottom_row])
    
    # Convert to PIL for annotations
    img = Image.fromarray(composite)
    
    if annotations:
        img = _add_text_annotations(img, annotations, layout='orthogonal')
    
    img.save(str(out_png))
    return str(out_png)

def _add_text_annotations(
    img: Image.Image,
    metadata: Dict[str, Any],
    layout: str = 'simple'
) -> Image.Image:
    """
    Add metadata text overlay to help VLM understand the image.
    
    Args:
        img: PIL Image
        metadata: Dict with keys like 'modality', 'format', 'shape', 'spacing', etc.
        layout: 'simple', 'orthogonal', or 'detailed'
    """
    # Create a copy to draw on
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Try to load a font, fall back to default
    try:
        # Try common system fonts
        font_size = max(12, min(20, img.height // 40))
        try:
            # Linux
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                # Windows
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Build annotation text
    lines = []
    
    if layout == 'orthogonal':
        lines.append("Top: MIP projections | Bottom: Middle slices")
        lines.append("Left: Axial | Center: Coronal | Right: Sagittal")
    
    # Add metadata
    if metadata.get('modality'):
        lines.append(f"Modality: {metadata['modality']}")
    if metadata.get('format'):
        lines.append(f"Format: {metadata['format']}")
    if metadata.get('shape'):
        shp = metadata['shape']
        if isinstance(shp, (list, tuple)):
            dim_str = f"{len(shp)}D {tuple(shp)}"
        else:
            dim_str = str(shp)
        lines.append(f"Dimensions: {dim_str}")
    if metadata.get('spacing'):
        lines.append(f"Spacing: {metadata['spacing']}")
    if metadata.get('note'):
        lines.append(f"Note: {metadata['note']}")
    
    # Draw semi-transparent background
    text_height = len(lines) * (font_size + 4)
    padding = 8
    bg_height = text_height + 2 * padding
    
    # Create semi-transparent overlay
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw background rectangle
    overlay_draw.rectangle(
        [(0, 0), (img.width, bg_height)],
        fill=(0, 0, 0, 180)  # Semi-transparent black
    )
    
    # Composite overlay
    annotated = Image.alpha_composite(annotated.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(annotated)
    
    # Draw text
    y_offset = padding
    for line in lines:
        draw.text((padding, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += font_size + 4
    
    return annotated

def _build_preview_for_vlm(image_paths: Optional[List[str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Build an enhanced preview image optimized for VLM analysis.
    
    Strategy:
    - 2D images: Add metadata annotations
    - 3D volumes: Create orthogonal multi-view composite with annotations
    - 4D data: Extract representative 3D volume, then multi-view
    - Medical images: Ensure proper intensity windowing
    
    Returns:
        (preview_path, metadata_text)
    """
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
            arr = np.asarray(data)
            ext = Path(p).suffix.lower()
            
            # Extract metadata for annotations
            annotation_meta = {
                'format': meta.get('format', ext.upper().lstrip('.')),
                'shape': shp,
            }
            
            # Try to extract modality from metadata or filename
            if 'modality' in meta:
                annotation_meta['modality'] = meta['modality']
            elif hasattr(meta, 'Modality'):
                annotation_meta['modality'] = meta.Modality
            
            # Extract spacing if available
            if 'zooms' in meta:
                zooms = meta['zooms']
                if len(zooms) >= 3:
                    annotation_meta['spacing'] = f"{zooms[0]:.2f}×{zooms[1]:.2f}×{zooms[2]:.2f}mm"
                elif len(zooms) == 2:
                    annotation_meta['spacing'] = f"{zooms[0]:.2f}×{zooms[1]:.2f}mm"

            # Handle true color images (H, W, 3/4) safely
            # For PNG/JPEG/WebP, (H,W,3/4) is almost certainly color → render with annotations
            if _is_rgb_like(arr.shape) and ext in {".png", ".jpg", ".jpeg", ".webp"}:
                out = tmpdir / "image_annotated.png"
                img_uint8 = _to_uint8_image(arr)
                img_pil = Image.fromarray(img_uint8)
                img_annotated = _add_text_annotations(img_pil, annotation_meta, layout='simple')
                img_annotated.save(str(out))
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
                        out = tmpdir / "image_annotated.png"
                        img_uint8 = _to_uint8_image(arr)
                        img_pil = Image.fromarray(img_uint8)
                        img_annotated = _add_text_annotations(img_pil, annotation_meta, layout='simple')
                        img_annotated.save(str(out))
                        return str(out), meta_text
                except Exception:
                    # If tags can't be read, prefer treating TIFF (H,W,3) as a stack
                    pass

            # 3D volumes: Create enhanced multi-view composite
            if len(shp) == 3:
                png_path = tmpdir / "orthogonal_views.png"
                try:
                    # Try orthogonal views first (best for VLM understanding)
                    create_orthogonal_views(arr, png_path, annotations=annotation_meta)
                    if png_path.exists():
                        log.info(f"Created orthogonal view composite for 3D volume {shp}")
                        return str(png_path), meta_text
                except Exception as e:
                    log.warning(f"Orthogonal views failed: {e}, falling back to contact sheet")
                    # Fallback to contact sheet
                    png_path = tmpdir / "slices_grid.png"
                    try:
                        contact_sheet_slices(arr, png_path, max_slices=36, grid_cols=6)
                        # Add annotations to contact sheet
                        img = Image.open(str(png_path))
                        img = _add_text_annotations(img, annotation_meta, layout='simple')
                        img.save(str(png_path))
                        if png_path.exists():
                            return str(png_path), meta_text
                    except Exception:
                        pass
                    
                    # Final fallback: MIP montage
                    try:
                        mip_montage(arr, png_path)
                        if png_path.exists():
                            return str(png_path), meta_text
                    except Exception:
                        pass

            # 4D data: Extract representative 3D volume (mean over time), then multi-view
            if len(shp) == 4:
                vol = np.asarray(data).mean(axis=-1)  # Average over 4th dimension
                annotation_meta['note'] = f"4D data: averaged over {shp[3]} timepoints"
                out = tmpdir / "orthogonal_4d.png"
                try:
                    create_orthogonal_views(vol, out, annotations=annotation_meta)
                    if out.exists():
                        log.info(f"Created orthogonal view for 4D volume {shp}")
                        return str(out), meta_text
                except Exception as e:
                    log.warning(f"4D orthogonal failed: {e}, trying gif")
                    # Fallback to animated GIF
                    out = tmpdir / "sweep.gif"
                    step = max(1, vol.shape[2] // 64)
                    slice_gif(vol, out, axis=2, step=step, fps=12)
                    return str(out), meta_text

            # 2D images: Add annotations
            if len(shp) == 2:
                out = tmpdir / "image_annotated.png"
                arr2 = _norm_uint8(arr)  # Use consistent normalization
                img_pil = Image.fromarray(arr2)
                img_annotated = _add_text_annotations(img_pil, annotation_meta, layout='simple')
                img_annotated.save(str(out))
                return str(out), meta_text
                
        except Exception as e:
            log.warning(f"Preview generation failed for {p}: {e}")
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