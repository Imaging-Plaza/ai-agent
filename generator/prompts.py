# generator/prompts.py
from __future__ import annotations
from typing import Iterable, List, Optional
from generator.schema import CandidateDoc, PerceptionCues


# ------------------------- Helpers -------------------------

def _csv(xs: Iterable[str]) -> str:
    return ", ".join([x for x in xs if x])

def _fmt_candidate(c: CandidateDoc) -> str:
    return (
        f"name={c.name} | "
        f"tasks=[{_csv(c.tasks)}] | "
        f"modality=[{_csv(c.modality)}] | "
        f"dims=[{_csv(c.dims)}] | "
        f"anatomy=[{_csv(c.anatomy)}] | "
        f"inputs=[{_csv(c.input_formats)}] | "
        f"outputs=[{_csv(c.output_types)}] | "
        f"language={c.language or ''} | "
        f"gpu_required={c.gpu_required if c.gpu_required is not None else ''} | "
        f"license={c.license or ''} | "
        f"demo={('hf_space=' + c.hf_space) if c.hf_space else ''}"
    ).strip()


# ===========================================================
# Legacy “Plan & Code” generator (kept for back-compat)
# ===========================================================

PLANNER_SYSTEM = (
    "You are an imaging software selection assistant and planner.\n"
    "Given the user's task, lightweight perception cues, and a small set of candidate tools, "
    "you must pick exactly ONE tool from the candidates and explain why.\n"
    "Then produce a SHORT, high-level execution plan and a SMALL Python code snippet that demonstrates "
    "how one could run or prototype the task locally (it may use placeholders if needed).\n"
    "\n"
    "Rules:\n"
    "- Only choose a tool from the candidate list. Never invent new tools.\n"
    "- Prefer candidates that match task, modality, dimensionality, I/O formats, and available demos.\n"
    "- Keep the rationale under 60 words.\n"
    "- Keep steps to 3–6 bullets, concise and actionable.\n"
    "- Code must be valid Python, minimal, and self-contained (imports allowed). If true execution is unclear, use a clear placeholder.\n"
    "- Respond with STRICT JSON ONLY. No prose, no markdown fences.\n"
    "\n"
    "JSON schema (keys and types MUST match):\n"
    "{\n"
    '  "choice": "string (must be one of the candidate names)",\n'
    '  "alternates": ["string", "... (up to 3)"],\n'
    '  "why": "string (<= 60 words)",\n'
    '  "steps": ["string", "... (3-6 items)"],\n'
    '  "code": "string (Python snippet)"\n'
    "}\n"
)

def build_user_prompt(
    user_task: str,
    cues: Optional[PerceptionCues],
    candidates: List[CandidateDoc],
    image_path: str = "",
    out_mask_path: str = "mask.nii.gz",
    overlay_png_path: str = "overlay.png",
) -> str:
    cue_lines: List[str] = []
    if cues:
        if cues.modality: cue_lines.append(f"modality={cues.modality}")
        if cues.dims:     cue_lines.append(f"dims={cues.dims}")
        if cues.anatomy:  cue_lines.append(f"anatomy={cues.anatomy}")
        if cues.task:     cue_lines.append(f"task={cues.task}")
        if cues.io_hint:  cue_lines.append(f"io_hint={cues.io_hint}")

    cand_block = "\n".join(f"- {_fmt_candidate(c)}" for c in candidates)

    prompt = (
        f"User task: {user_task}\n"
        f"Perception cues: {', '.join(cue_lines) if cue_lines else '(none)'}\n"
        f"Input image path (for context only, not to read): {image_path or '(not provided)'}\n"
        f"Output mask path suggestion: {out_mask_path}\n"
        f"Overlay image path suggestion: {overlay_png_path}\n"
        f"Candidates (choose exactly one by name):\n{cand_block}\n"
        "Return STRICT JSON only (no markdown)."
    )
    return prompt


# ===========================================================
# Minimal “Selection-only” generator (recommended for linking)
# ===========================================================

SELECTOR_SYSTEM = (
    "You are a software selector. From the provided candidate imaging tools, "
    "pick exactly ONE best tool for the user's task. Do not invent tools.\n"
    "\n"
    "Selection criteria (in order):\n"
    "1) Task fit (e.g., segmentation, deblurring)\n"
    "2) Modality (CT/MRI/XR/US/natural) and dimensionality (2D/3D)\n"
    "3) I/O compatibility (supported input formats, available outputs)\n"
    "4) Practicality (language, GPU requirement) if relevant\n"
    "\n"
    "Output must be STRICT JSON ONLY with EXACTLY these keys:\n"
    "{\n"
    '  "choice": "string (one of the candidate names)",\n'
    '  "alternates": ["string", "... up to 3"],\n'
    '  "why": "string (<= 60 words)"\n'
    "}\n"
    "No extra keys. No markdown. No code."
)

def build_selector_prompt(
    user_task: str,
    cues: Optional[PerceptionCues],
    candidates: List[CandidateDoc],
    image_path: str = "",
) -> str:
    cue_lines: List[str] = []
    if cues:
        if cues.modality: cue_lines.append(f"modality={cues.modality}")
        if cues.dims:     cue_lines.append(f"dims={cues.dims}")
        if cues.anatomy:  cue_lines.append(f"anatomy={cues.anatomy}")
        if cues.task:     cue_lines.append(f"task={cues.task}")
        if cues.io_hint:  cue_lines.append(f"io_hint={cues.io_hint}")

    cand_block = "\n".join(f"- {_fmt_candidate(c)}" for c in candidates)

    prompt = (
        f"User task: {user_task}\n"
        f"Perception cues: {', '.join(cue_lines) if cue_lines else '(none)'}\n"
        f"Input image path (context only): {image_path or '(not provided)'}\n"
        f"Candidates (choose exactly one by name):\n{cand_block}\n"
        "Return STRICT JSON only with keys: choice, alternates, why."
    )
    return prompt
