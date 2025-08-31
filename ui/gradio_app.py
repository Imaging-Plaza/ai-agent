# ui app built via gradio
from __future__ import annotations

# --- sys.path so local packages import cleanly --------------------------------
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- env ----------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

# --- logging (console + rotating file, DEBUG via DEBUG=1) ---------------------
import logging, logging.handlers
from datetime import datetime
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

debug_on = str(os.getenv("DEBUG", "0")).lower() in ("1", "true", "yes", "on")
console_level = os.getenv("LOGLEVEL_CONSOLE", "INFO").upper()
file_level = os.getenv("LOGLEVEL_FILE", "DEBUG" if debug_on else "INFO").upper()
file_log_enabled = str(os.getenv("FILE_LOG", "1")).lower() in ("1", "true", "yes", "on")

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
from typing import Optional, List, Dict
import gradio as gr

from retriever.embedders import SoftwareDoc
from api.pipeline import RAGImagingPipeline

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
            docs.append(SoftwareDoc.model_validate(o))
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
            docs.append(SoftwareDoc.model_validate(o))
        except Exception as e:
            snippet = (line[:160] + "…") if len(line) > 160 else line
            raise ValueError(f"Invalid catalog JSON on line {i}:\n{snippet}\n\nError: {e}")

    if not docs:
        raise ValueError("Catalog is empty.")
    return docs

# --- pipeline singleton + in-memory doc index --------------------------------
_pipe: Optional[RAGImagingPipeline] = None
_DOCS: List[SoftwareDoc] = []
_DOC_BY_NAME: Dict[str, SoftwareDoc] = {}

def get_pipeline() -> RAGImagingPipeline:
    global _pipe, _DOCS, _DOC_BY_NAME
    if _pipe is None:
        _DOCS = _load_catalog(CATALOG_PATH)
        _DOC_BY_NAME = {d.name: d for d in _DOCS if getattr(d, "name", None)}
        log.info("Loaded %d tools from %s", len(_DOCS), CATALOG_PATH)
        _pipe = RAGImagingPipeline(docs=_DOCS, index_dir="artifacts/rag_index")
        log.info("Pipeline ready")
    return _pipe

# --- helpers ------------------------------------------------------------------
def _coerce_gradio_files_to_paths(fobjs) -> List[str]:
    """
    Gradio 'Files' returns a list where each item can be a str (path)
    or a dict with 'name'/'path'.
    """
    out: List[str] = []
    if not fobjs:
        return out
    for f in fobjs:
        if isinstance(f, str):
            out.append(f)
        elif isinstance(f, dict):
            p = f.get("name") or f.get("path")
            if p:
                out.append(p)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped

def _fmt_toolcard(name: str) -> str:
    """
    Render a compact toolcard for a tool name using the loaded catalog.
    """
    d = _DOC_BY_NAME.get(name)
    if not d:
        return f"- **{name}**"
    modality = ", ".join(d.modality) if getattr(d, "modality", None) else ""
    dims = " / ".join(f"{x}D" for x in getattr(d, "dims", []) or [])
    license_ = getattr(d, "license", "") or ""
    tags = []
    if getattr(d, "tasks", None): tags.extend(d.tasks)
    if getattr(d, "keywords", None): tags.extend(d.keywords)
    tags = ", ".join(sorted(set(t for t in tags if t)))[:160]
    bits = []
    if modality: bits.append(modality)
    if dims: bits.append(dims)
    if license_: bits.append(f"License: {license_}")
    meta = " • ".join(bits)
    desc = getattr(d, "description", "") or ""
    short = (desc[:160] + "…") if len(desc) > 160 else desc
    # markdown card
    out = f"**{d.name}**"
    if meta: out += f"  \n{meta}"
    if short: out += f"  \n_{short}_"
    if tags: out += f"  \n`{tags}`"
    return out

# --- main action (streaming with progress messages) ---------------------------
def run_agent(task_text: str, uploaded_files, history_rows):
    """
    Streamed return (7-tuple):
    (choice_md, link_md, why_md, toolcards_md, history_df, demo_link_text, status_msg)
    """
    # default outputs helper
    def _blank():
        return "", "", "", "", history_rows, "", ""

    if not task_text:
        yield _blank()
        return

    image_paths = _coerce_gradio_files_to_paths(uploaded_files)
    for p in image_paths:
        if not os.path.exists(p) and not os.path.isdir(p):
            yield "", "", f"❌ Could not read uploaded path: {p}", "", history_rows, "", "Invalid path"
            return

    # Step 1: announce start
    yield "⏳ Analyzing request…", "_Preparing…_", "", "", history_rows, "", "Starting…"
    pipe = get_pipeline()

    # Step 2: retrieval stage (text-only + optional format tokens from inputs)
    yield "⏳ Retrieving candidates…", "_Searching & reranking…_", "", "", history_rows, "", "Retrieving…"

    # Step 3: selection (single VLM call; pipeline builds preview)
    result = pipe.recommend_and_link(image_paths=image_paths, user_task=task_text)
    if "error" in result:
        yield "", "", f"❌ {result['error']}", "", history_rows, "", "Done."
        return

    # Final: present results
    choice = result.get("choice", "")
    choice_md = f"**Selected software:** `{choice}`" if choice else "**Selected software:** _none_"

    demo_link = result.get("demo_link") or ""
    link_md = (
        f"[Open runnable demo]({demo_link})" if demo_link else "_No runnable demo available for this tool._"
    )

    why_md = result.get("why", "") or ""
    scores = result.get("scores") or {}
    alternates = result.get("alternates", [])

    # Confidence readout
    if scores:
        top = float(scores.get("top", 0.0))
        second = float(scores.get("second", 0.0))
        margin = float(scores.get("margin", 0.0))
        why_md += (
            f"\n\nConfidence: top={top:.3f}, second={second:.3f}, margin={margin:.3f}"
        )

    # Toolcards for chosen + alternates
    cards: List[str] = []
    if choice:
        cards.append(_fmt_toolcard(choice))
    for a in alternates[:3]:
        cards.append(_fmt_toolcard(a))
    toolcards_md = "\n\n---\n\n".join(cards) if cards else ""

    # Update history (session-local)
    from pandas import DataFrame
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [ts, task_text[:80], choice, "yes" if demo_link else "no"]

    # Gradio may pass a DataFrame back; normalize to list-of-lists
    if isinstance(history_rows, DataFrame):
        history_rows = history_rows.values.tolist()
    elif history_rows is None:
        history_rows = []

    new_history = history_rows + [row]

    # status
    status = "Done."
    yield choice_md, link_md, why_md, toolcards_md, new_history, demo_link, status

def reset_all(history_rows):
    # task_box, images_in, out_choice, out_link, out_why, out_cards, history_df, copy_link_tb, copy_tip
    return "", None, "", "", "", "", history_rows, "", ""


def clear_history():
    return []

# --- UI -----------------------------------------------------------------------
with gr.Blocks(title="AI Imaging Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 🔎 AI-assisted software picker\n"
        "Describe your task and (optionally) drop one or more images/volumes. "
        "We’ll select the best software and send you to its public runnable demo (when available)."
    )

    with gr.Row():
        task_box = gr.Textbox(
            label="Describe your task",
            placeholder='e.g., "Segment the lungs" or "Deblur this photo"',
            lines=2,
        )

    images_in = gr.Files(
        label="Images / volumes (drag & drop multiple; DICOM zip/folder, NIfTI, TIFF, PNG/JPEG, etc.)",
        file_count="multiple",
        type="filepath",
        file_types=[
            ".zip", ".dcm",
            ".nii", ".nii.gz",
            ".tif", ".tiff",
            ".png", ".jpg", ".jpeg", ".bmp", ".webp",
        ],
    )

    with gr.Row():
        run_btn = gr.Button("Find software", variant="primary")
        reset_btn = gr.Button("Reset", variant="secondary")

    # Outputs
    out_choice = gr.Markdown()
    out_link = gr.Markdown()
    out_why = gr.Markdown()
    out_cards = gr.Markdown()

    # History
    history_state = gr.State(value=[])
    with gr.Accordion("Result history (session)", open=False):
        history_df = gr.Dataframe(
            headers=["time", "task (head)", "choice", "demo?"],
            datatype=["str", "str", "str", "str"],
            row_count=(0, "dynamic"),
            col_count=(4, "fixed"),
            interactive=False,
            wrap=True,
            label="History",
            elem_id="history_df",
        )
        with gr.Row():
            clear_btn = gr.Button("Clear history")

    # Demo link textbox with native copy button
    copy_link_tb = gr.Textbox(
        label="Demo link",
        value="",
        interactive=False,
        show_copy_button=True,
    )
    copy_tip = gr.Markdown("")

    # Streamed outputs
    run_btn.click(
        fn=run_agent,
        inputs=[task_box, images_in, history_state],
        outputs=[out_choice, out_link, out_why, out_cards, history_df, copy_link_tb, copy_tip],
        show_progress="full",
        api_name="recommend_and_link",
    ).then(
        fn=lambda rows: rows,
        inputs=history_df,
        outputs=history_state,
        show_progress=False,
    )

    # Clear history
    clear_btn.click(
        fn=lambda: (clear_history(), []),
        inputs=None,
        outputs=[history_df, history_state],
        show_progress=False,
    )

    reset_btn.click(
        fn=reset_all,
        inputs=[history_state],   # pass current history in
        outputs=[task_box, images_in, out_choice, out_link, out_why, out_cards, history_df, copy_link_tb, copy_tip],
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.queue(api_open=False).launch(
        server_name="127.0.0.1",                 # local only (no public URL)
        server_port=int(os.getenv("PORT", "7860")),
        inbrowser=True,
        show_error=True,
    )
