# ui app built via gradio
from __future__ import annotations

# --- sys.path so local packages import cleanly --------------------------------
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- env ----------------------------------------------------------------------
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# --- logging (console + rotating file, DEBUG via DEBUG=1) ---------------------
import logging, logging.handlers
from datetime import datetime
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

debug_on = str(os.getenv("DEBUG", "0")).lower() in ("1","true","yes","on")
console_level = os.getenv("LOGLEVEL_CONSOLE", "INFO").upper()
file_level = os.getenv("LOGLEVEL_FILE", "DEBUG" if debug_on else "INFO").upper()
file_log_enabled = str(os.getenv("FILE_LOG", "1")).lower() in ("1","true","yes","on")

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
root = logging.getLogger()
root.handlers.clear()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(console_level)
ch.setFormatter(fmt)
root.addHandler(ch)

if file_log_enabled:
    logfile = os.path.join(LOG_DIR, f"app_{datetime.now():%Y%m%d}.log")
    fh = logging.handlers.TimedRotatingFileHandler(
        logfile, when="midnight", backupCount=14, encoding="utf-8"
    )
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

log = logging.getLogger("ui")
log.info("Starting Gradio UI")

# --- imports that rely on sys.path -------------------------------------------
from pathlib import Path
from typing import Optional, Any, List
import gradio as gr

from retriever.embedders import SoftwareDoc
from api.pipeline import RAGImagingPipeline  # exposes recommend_and_link()

# --- config -------------------------------------------------------------------
CATALOG_PATH = os.getenv("SOFTWARE_CATALOG", "data/sample.jsonl")
WORKDIR = os.getenv("RAG_WORKDIR", "runs")
Path(WORKDIR).mkdir(parents=True, exist_ok=True)

# --- catalog loader (supports JSON or JSONL) ----------------------------------
def _load_catalog(path: str) -> List[SoftwareDoc]:
    import json
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Software catalog not found at {p.resolve()}.\n"
            "Set SOFTWARE_CATALOG or create data/sample.jsonl"
        )

    text = p.read_text(encoding="utf-8").strip()
    docs: List[SoftwareDoc] = []

    # Try full JSON first
    try:
        obj = json.loads(text)
        obj = [obj] if isinstance(obj, dict) else obj
        for o in obj:
            try:
                docs.append(SoftwareDoc.model_validate(o))  # pydantic v2
            except Exception:
                docs.append(SoftwareDoc.parse_obj(o))       # pydantic v1
        return docs
    except Exception:
        pass

    # Fallback: JSONL
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            try:
                docs.append(SoftwareDoc.model_validate(o))
            except Exception:
                docs.append(SoftwareDoc.parse_obj(o))
        except Exception as e:
            snippet = (line[:160] + "…") if len(line) > 160 else line
            raise ValueError(f"Invalid catalog JSON on line {i}:\n{snippet}\n\nError: {e}")

    if not docs:
        raise ValueError("Catalog is empty.")
    return docs

# --- pipeline singleton -------------------------------------------------------
_pipe: Optional[RAGImagingPipeline] = None
def get_pipeline() -> RAGImagingPipeline:
    global _pipe
    if _pipe is None:
        docs = _load_catalog(CATALOG_PATH)
        log.info("Loaded %d tools from %s", len(docs), CATALOG_PATH)
        _pipe = RAGImagingPipeline(docs, workdir=WORKDIR)
        log.info("Pipeline ready")
    return _pipe

# --- helpers ------------------------------------------------------------------
def _coerce_gradio_file_to_path(fobj) -> Optional[str]:
    """
    Gradio File can be a str (path) or dict with 'name'/'path' depending on version.
    """
    if not fobj:
        return None
    if isinstance(fobj, str):
        return fobj
    if isinstance(fobj, dict):
        return fobj.get("name") or fobj.get("path")
    return None

# --- main action (streaming with progress messages) ---------------------------
def run_agent(task_text: str, image_file):
    if not task_text:
        yield "", "", "Please describe your task."
        return

    img_path = _coerce_gradio_file_to_path(image_file)
    if img_path and not os.path.exists(img_path):
        yield "", "", "Could not read the uploaded file path."
        return

    # Step 1: announce start
    yield "⏳ Analyzing request…", "_Preparing…_", ""
    pipe = get_pipeline()

    # Step 2: retrieval stage (text-only + optional format token)
    yield "⏳ Retrieving candidates…", "_Searching & reranking…_", ""

    # Step 3: selection (single VLM call happens inside the pipeline)
    result = pipe.recommend_and_link(image_path=img_path, user_task=task_text)
    if "error" in result:
        yield "", "", f"❌ {result['error']}"
        return

    # Final: present results (no data return, only link out)
    choice_md = f"**Selected software:** `{result['choice']}`"
    link_md = (
        f"[Open runnable demo]({result['demo_link']})"
        if result.get("demo_link")
        else "_No runnable demo available for this tool._"
    )

    why_md = result.get("why", "")
    scores = result.get("scores")
    if scores:
        why_md += (
            f"\n\nConfidence: top={scores['top']:.3f}, "
            f"second={scores['second']:.3f}, margin={scores['margin']:.3f}"
        )

    yield choice_md, link_md, (why_md or "")

def reset_all():
    return "", None, "", "", ""

# --- UI -----------------------------------------------------------------------
with gr.Blocks(title="AI Imaging Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 🔎 AI-assisted software picker\n"
        "Describe your task and (optionally) drop an image. "
        "We’ll select the best software and send you to its public runnable demo (when available)."
    )

    with gr.Row():
        task_box = gr.Textbox(
            label="Describe your task",
            placeholder='e.g., "Segment the lungs" or "Deblur this photo"',
            lines=2,
        )
    image_in = gr.File(
        label="Image (drag & drop)",
        file_count="single",
        file_types=[
            ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp",
            ".nii", ".nii.gz"
        ],
    )

    with gr.Row():
        run_btn = gr.Button("Find software", variant="primary")
        reset_btn = gr.Button("Reset", variant="secondary")

    out_choice = gr.Markdown()
    out_link = gr.Markdown()
    out_why = gr.Markdown()

    # Streamed outputs give a visible “loader” feeling in the UI
    run_btn.click(
        fn=run_agent,
        inputs=[task_box, image_in],
        outputs=[out_choice, out_link, out_why],
        show_progress="full",
        api_name="recommend_and_link",
    )
    reset_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[task_box, image_in, out_choice, out_link, out_why],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(
        server_name="127.0.0.1",                 # local only (no public URL)
        server_port=int(os.getenv("PORT", "7860")),
        inbrowser=True,
        show_error=True,
    )
