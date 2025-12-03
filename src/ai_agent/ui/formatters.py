import os
from typing import Tuple, Optional

from ai_agent.retriever.software_doc import SoftwareDoc


def format_tool_card(doc: SoftwareDoc, accuracy: float, why: str, rank: int) -> str:
    """Format a tool recommendation as a rich card."""
    modality = ", ".join(doc.modality) if getattr(doc, "modality", None) else ""
    dims = " / ".join(f"{x}D" for x in (getattr(doc, "dims", None) or []))
    license_ = getattr(doc, "license", "") or ""
    
    tags = []
    if getattr(doc, "tasks", None):
        tags.extend(doc.tasks)
    if getattr(doc, "keywords", None):
        tags.extend(doc.keywords)
    tags_str = ", ".join(sorted(set(t for t in tags if t)))[:160]
    
    desc = getattr(doc, "description", "") or ""
    short_desc = (desc[:200] + "…") if len(desc) > 200 else desc
    
    card = f"### {rank}. {doc.name} — {accuracy:.1f}% match\n\n"
    
    meta_parts = []
    if modality: meta_parts.append(f"**Modality:** {modality}")
    if dims: meta_parts.append(f"**Dimensions:** {dims}")
    if license_: meta_parts.append(f"**License:** {license_}")
    if meta_parts:
        card += " • ".join(meta_parts) + "\n\n"
    
    if short_desc:
        card += f"_{short_desc}_\n\n"
    
    if why:
        card += f"**Why:** {why}\n\n"
    
    if tags_str:
        card += f"**Tags:** `{tags_str}`\n\n"
    
    return card


def format_file_preview(file_path: str) -> Tuple[str, Optional[str]]:
    """
    Create a preview for uploaded files.
    Returns (description_text, preview_image_path)
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    basename = os.path.basename(file_path)
    
    # Image formats - can be displayed directly
    if ext in ('png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'):
        return f"📷 {basename}", file_path
    
    # TIFF might be multi-page
    if ext in ('tif', 'tiff'):
        return f"🖼️ {basename} (TIFF stack)", file_path
    
    # Volume formats
    if ext in ('nii', 'dcm') or file_path.endswith('.nii.gz'):
        return f"🧠 {basename} (medical volume)", None
    
    # Data formats
    if ext == 'csv':
        return f"📊 {basename} (CSV data)", None
    
    if ext in ('json', 'jsonl'):
        return f"📋 {basename} (JSON)", None
    
    if ext == 'xml':
        return f"📄 {basename} (XML)", None
    
    # Media formats
    if ext in ('mp3', 'wav', 'ogg'):
        return f"🎵 {basename} (audio)", None
    
    if ext in ('mp4', 'avi', 'mov', 'webm'):
        return f"🎬 {basename} (video)", None
    
    # Generic file
    return f"📎 {basename}", None
