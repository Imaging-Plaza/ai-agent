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
    if modality:
        meta_parts.append(f"**Modality:** {modality}")
    if dims:
        meta_parts.append(f"**Dimensions:** {dims}")
    if license_:
        meta_parts.append(f"**License:** {license_}")
    if meta_parts:
        card += " • ".join(meta_parts) + "\n\n"

    if short_desc:
        card += f"_{short_desc}_\n\n"

    if why:
        card += f"**Why:** {why}\n\n"

    if tags_str:
        card += f"**Tags:** `{tags_str}`\n\n"

    return card

