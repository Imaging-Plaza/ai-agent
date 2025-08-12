# generator/prompts.py
from __future__ import annotations
from typing import List
from generator.schema import PerceptionCues, CandidateDoc

PLANNER_SYSTEM = """You recommend medical imaging software and emit a short runnable Python demo.
Hard constraints: prefer matches for modality, dims, input_formats, gpu_required, language.
Be concise. Output JSON only, as {choice, alternates, why, steps, code}."""

def format_candidates(cands: List[CandidateDoc]) -> str:
    rows = []
    for c in cands:
        rows.append({
            "name": c.name,
            "repo_url": c.repo_url,
            "tasks": c.tasks,
            "modality": c.modality,
            "dims": c.dims,
            "anatomy": c.anatomy,
            "input_formats": c.input_formats,
            "output_types": c.output_types,
            "language": c.language,
            "install_cmd": c.install_cmd,
            "weights_available": c.weights_available,
            "license": c.license,
            "gpu_required": c.gpu_required,
            "sample_snippet": c.sample_snippet,
        })
    # Simple JSONL-like string; your caller can pass actual JSON if using tools mode.
    import json
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)

def build_user_prompt(
    user_task: str,
    cues: PerceptionCues | None,
    candidates: List[CandidateDoc],
    image_path: str,
    out_mask_path: str,
    overlay_png_path: str,
) -> str:
    cues_json = cues.model_dump() if cues else {}
    return f"""Task: {user_task}
Perception: {cues_json}

Candidates (JSONL):
{format_candidates(candidates)}

Return strict JSON with:
{{
  "choice": "<best tool name>",
  "alternates": ["<name1>", "<name2>"],
  "why": "<cite matching fields>",
  "steps": ["<<= 4 bullets>"],
  "code": "<= 25 lines, Python. Install via candidate install_cmd if needed. \
Load image from '{image_path}', write a mask to '{out_mask_path}', and save an overlay PNG to '{overlay_png_path}'. \
Avoid GUIs; only file I/O.>"
}}
"""
