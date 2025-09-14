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
from pandas import DataFrame
import json

from retriever.embedders import SoftwareDoc
from api.pipeline import RAGImagingPipeline
from utils.file_validator import FileValidator  # <-- Add to imports section

# --- config -------------------------------------------------------------------
CATALOG_PATH = os.getenv("SOFTWARE_CATALOG", "data/sample.jsonl")

# --- catalog loader (supports JSON or JSONL) ----------------------------------
def _load_catalog(path: str) -> List[SoftwareDoc]:
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
    """Process query and return recommendations"""
    if not task_text:
        yield _blank()
        return

    # Validate uploaded files
    image_paths = _coerce_gradio_files_to_paths(uploaded_files)
    if image_paths:
        valid_paths, errors = FileValidator.validate_files(image_paths)
        # File validation error yield
        if errors:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = [ts, task_text[:80], "none", "no"]  # Add to history
            new_history = (history_rows or []) + [row]
            
            yield (
                "❌ **File validation failed:**\n\n" + "\n".join(f"- {err}" for err in errors),
                "",  # link_md
                "",  # why_md 
                "",  # toolcards_md
                new_history,  # history
                ""   # demo_link
            )
            return
        image_paths = valid_paths

    # Step 1: announce start
    yield "⏳ Analyzing request…", "_Preparing…_", "", "", history_rows, ""

    pipe = get_pipeline()

    # Step 2: run pipeline
    result = pipe.recommend_and_link(image_paths=image_paths, user_task=task_text)
    # Update error yield
    if "error" in result:
        yield "", "", f"❌ {result['error']}", "", history_rows, ""
        return

    # Handle no tools case with explanation
    if not result.get("choices"):
        reason = result.get("reason", "Unknown reason")
        explanation = result.get("explanation")
        
        # Add to history with reason
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [ts, task_text[:80], f"No tool ({reason})", "no"]
        
        if isinstance(history_rows, DataFrame):
            history_rows = history_rows.values.tolist()
        elif history_rows is None:
            history_rows = []
            
        new_history = history_rows + [row]
        
        # Format message with reason and explanation
        message = (
            "❌ **No Suitable Tools Found**\n\n"
            f"**Reason**: {reason}\n\n"
        )
        
        if explanation:  # Only add explanation if it exists
            message += f"**Details**: {explanation}\n\n"
            
        message += "_Consider refining your request or checking if your image format is supported._"
        
        yield (
            message,     # choice_md
            "",         # link_md
            "",         # why_md
            "",         # toolcards_md
            new_history,
            ""          # demo_link
        )
        return

    # Create results table with accuracy threshold warning
    choices_table = ""
    low_accuracy_warning = False
    
    for choice in result.get("choices", []):
        if choice.get("accuracy", 0) < 50:
            low_accuracy_warning = True
            break
    
    if low_accuracy_warning:
        choices_table = "⚠️ **Warning**: Some suggested tools have low accuracy scores (<50%). Consider reviewing alternatives.\n\n"
    
    choices_table += "| Rank | Tool | Accuracy | Explanation | Demo |\n"
    choices_table += "|------|------|----------|-------------|------|\n"
    
    demo_link = ""  # Store first demo link for the copy button
    
    for choice in result.get("choices", []):
        name = choice["name"]
        rank = choice["rank"]
        accuracy = f"{choice.get('accuracy', 0):.1f}%"
        why = choice["why"]
        demo = f"[Open demo]({choice.get('demo_link', '')})" if choice.get('demo_link') else "_No demo_"
        
        # Store first demo link
        if rank == 1 and choice.get('demo_link'):
            demo_link = choice['demo_link']
            
        choices_table += f"| {rank} | `{name}` | {accuracy} | {why} | {demo} |\n"

    # Format tool cards
    cards = []
    for choice in result.get("choices", []):
        cards.append(_fmt_toolcard(choice["name"]))
    toolcards_md = "\n\n---\n\n".join(cards) if cards else ""

    # Update history
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_choice = result.get("choices", [{}])[0].get("name", "none")
    has_demo = "yes" if demo_link else "no"
    row = [ts, task_text[:80], top_choice, has_demo]

    if isinstance(history_rows, DataFrame):
        history_rows = history_rows.values.tolist()
    elif history_rows is None:
        history_rows = []

    new_history = history_rows + [row]

    # Show confidence scores if available
    scores = result.get("scores", {})
    if scores:
        confidence = (
            f"\n\n**Confidence metrics:**  \n"
            f"Top score: {scores.get('top', 0):.3f}  \n"
            f"Second score: {scores.get('second', 0):.3f}  \n"
            f"Margin: {scores.get('margin', 0):.3f}"
        )
    else:
        confidence = ""

    # Return results
    # Final yield - remove status from tuple
    yield (
        choices_table,     # choice_md now contains the table
        "",               # link_md (empty since links are in table)
        confidence,       # why_md shows confidence metrics
        toolcards_md,     # detailed tool cards
        new_history,
        demo_link         # for copy button
    )

def reset_all(history_rows):
    # task_box, images_in, out_choice, out_link, out_why, out_cards, history_df, copy_link_tb
    return "", None, "", "", "", "", history_rows, ""


def clear_history():
    return []

def _blank():
    """Default empty return values"""
    return "", "", "", "", [], ""  # 6 values matching the output components

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
        file_types=None
    )

    with gr.Row():
        run_btn = gr.Button("Find software", variant="primary")
        reset_btn = gr.Button("Reset", variant="secondary")

    # Outputs
    out_choice = gr.Markdown()
    out_link   = gr.Markdown()
    out_why    = gr.Markdown(visible=False)    # HIDDEN: confidence metrics
    out_cards  = gr.Markdown(visible=False)    # HIDDEN: alternatives/tool cards

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
        outputs=[
            out_choice,    # choice_md
            out_link,      # link_md
            out_why,       # why_md
            out_cards,     # toolcards_md
            history_df,    # history
            copy_link_tb   # demo_link
        ],
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

    # Reset button
    reset_btn.click(
        fn=reset_all,
        inputs=[history_state],
        outputs=[task_box, images_in, out_choice, out_link, out_why, out_cards, history_df, copy_link_tb],
        show_progress="hidden",
    )

# RUN THE APP
def _bind_host() -> str:
    if os.getenv("BIND_HOST"):
        return os.getenv("BIND_HOST")

    in_docker = os.path.exists("/.dockerenv")
    return "0.0.0.0" if in_docker else "127.0.0.1"

def launch():
    host = _bind_host()
    port = int(os.getenv("PORT", "7860"))
    demo.queue(api_open=False).launch(
        server_name=host,
        server_port=port,
        inbrowser=False,
        show_error=True,
    )

if __name__ == "__main__":
    launch()
