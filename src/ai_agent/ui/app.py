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
from typing import Optional, List, Dict, Tuple
import json

import gradio as gr
from pandas import DataFrame

from retriever.embedders import SoftwareDoc
from api.pipeline import RAGImagingPipeline

from utils.file_validator import FileValidator
from utils.tags import strip_tags, parse_exclusions
from utils.previews import _build_preview_for_vlm

# --- config -------------------------------------------------------------------
CATALOG_PATH = os.getenv("SOFTWARE_CATALOG", "data/sample.jsonl")
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/rag_index")

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

    # Try full JSON array/object first
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
        _pipe = RAGImagingPipeline(docs=_DOCS, index_dir=INDEX_DIR)
        log.info("Pipeline ready")
    return _pipe

# --- helpers ------------------------------------------------------------------
def _coerce_gradio_files_to_paths(fobjs) -> List[str]:
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


def _clear_textbox():
    return ""


def _fmt_toolcard(name: str) -> str:
    d = _DOC_BY_NAME.get(name)
    if not d:
        return f"- **{name}**"
    modality = ", ".join(d.modality) if getattr(d, "modality", None) else ""
    dims = " / ".join(f"{x}D" for x in (getattr(d, "dims", None) or []))
    license_ = getattr(d, "license", "") or ""
    tags = []
    if getattr(d, "tasks", None):
        tags.extend(d.tasks)
    if getattr(d, "keywords", None):
        tags.extend(d.keywords)
    tags = ", ".join(sorted(set(t for t in tags if t)))[:160]
    bits = []
    if modality: bits.append(modality)
    if dims: bits.append(dims)
    if license_: bits.append(f"License: {license_}")
    meta = " • ".join(bits)
    desc = getattr(d, "description", "") or ""
    short = (desc[:160] + "…") if len(desc) > 160 else desc
    out = f"**{d.name}**"
    if meta: out += f"  \n{meta}"
    if short: out += f"  \n_{short}_"
    if tags: out += f"  \n`{tags}`"
    return out


def _choices_table_md(choices: List[dict]) -> str:
    if not choices:
        return ""
    rows = ["| # | Tool | Score | Notes |", "|---:|---|:----:|---|"]
    for c in choices:
        name = c.get("name", "?")
        acc = f"{float(c.get('accuracy', 0.0)):.1f}%"
        why = str(c.get("why", ""))
        short = (why[:120] + "…") if len(why) > 120 else why
        if c.get("demo_link"):
            name = f"{name} (🔗)"
        rows.append(f"| {int(c.get('rank', 0))} | {name} | {acc} | {short} |")
    return "\n".join(rows)

# --- validation ---------------------------------------------------------------
def _validate_files(paths: List[str]) -> Tuple[bool, str]:
    if not paths:
        return True, ""
    try:
        valid_paths, errors = FileValidator.validate_files(paths)  # type: ignore[attr-defined]
        if errors:
            if isinstance(errors, (list, tuple)):
                issues_text = "\n".join(f"• {x}" for x in errors)
            else:
                issues_text = str(errors)
            return False, f"One or more files look problematic:\n{issues_text}"
        
    except Exception as e:
        log.debug("FileValidator unavailable or raised: %r", e)
    return True, ""

# --- Chat handler (streaming + status/disable) --------------------------------
def _make_handler():
    def handle_message(message: str,
                       chat_history,
                       files,
                       history_rows,
                       conv_history,
                       ):
        # normalize inputs
        if isinstance(history_rows, DataFrame):
            history_rows = history_rows.values.tolist()
        history_rows = history_rows or []
        chat_history = chat_history or []
        conv_history = conv_history or []

        empty_radio = gr.update(choices=[], value=None)
        status_idle   = gr.update(value=None, visible=False)

        # Allow control-tag-only refine messages to proceed (don't treat as empty)
        raw_message = (message or "")
        has_any_text = bool(raw_message.strip())
        has_files    = bool(files)

        if (not has_any_text) and (not has_files):
            # nothing to do → DO NOT disable inputs; just return quietly
            yield (chat_history, history_rows, "", "", conv_history,
                   empty_radio, "—",
                   gr.update(visible=False),             # preview accordion hidden
                   gr.update(value=None, visible=False), # preview img hidden
                   gr.update(visible=False),             # refine accordion hidden
                   [],                                   # excluded_names
                   status_idle,
                   gr.update(interactive=True),          # re-enable msg
                   gr.update(interactive=True),          # re-enable submit
                   gr.update(interactive=True))          # re-enable files
            return

        # What the user sees in the chat (tags removed)
        visible_msg = strip_tags(raw_message)
        if not visible_msg:
            # If message had only control tags, show a friendly stub
            excluded = parse_exclusions(raw_message)
            suffix = f" (excluding: {', '.join(excluded)})" if excluded else ""
            visible_msg = f"Find alternatives{suffix}"

        # Now disable inputs and continue
        disable_inputs = (
            gr.update(interactive=False),  # msg
            gr.update(interactive=False),  # submit
            gr.update(interactive=False),  # files
        )

        # 0) immediately show user's message + show status and disable inputs
        chat_history = chat_history + [[visible_msg, None]]
        conv_history = conv_history + [f"User: {visible_msg}"]
        status = gr.update(value="🔄 Validating files…", visible=True)
        yield (chat_history, history_rows, "", "", conv_history,
               empty_radio, "—",
               gr.update(visible=False),                # preview acc
               gr.update(value=None, visible=False),    # preview img
               gr.update(visible=False),                # refine acc
               [],                                      # excluded_names
               status,
               *disable_inputs)

        # 1) file validation
        paths = _coerce_gradio_files_to_paths(files)
        ok, why_not = _validate_files(paths)
        if not ok:
            chat_history[-1][1] = f"⚠️ File issues detected.\n\n{why_not}"
            status_done = gr.update(value="⚠️ File validation failed.", visible=True)
            # re-enable inputs
            enable_inputs = (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
            yield (chat_history, history_rows, "", "", conv_history,
                   empty_radio, "—",
                   gr.update(visible=False),
                   gr.update(value=None, visible=False),
                   gr.update(visible=False),
                   [],                                      # excluded_names
                   status_done,
                   *enable_inputs)
            return

        # 2) build preview (collapsed & only visible if present)
        status = gr.update(value="🖼️ Building preview…", visible=True)
        yield (chat_history, history_rows, "", "", conv_history,
               empty_radio, "—",
               gr.update(visible=False),
               gr.update(value=None, visible=False),
               gr.update(visible=False),
               [],                                      # excluded_names
               status,
               *disable_inputs)

        preview_path = None
        try:
            preview_path, _meta_text = _build_preview_for_vlm(paths)
        except Exception:
            preview_path = None

        preview_acc_upd = gr.update(visible=bool(preview_path))
        preview_img_upd = gr.update(value=preview_path, visible=bool(preview_path))

        yield (chat_history, history_rows, "", "", conv_history,
               empty_radio, "—",
               preview_acc_upd,
               preview_img_upd,
               gr.update(visible=False),                  # refine hidden while working
               [],                                        # excluded_names
               gr.update(value="📚 Reranking candidates…", visible=True),
               *disable_inputs)

        # 3) run pipeline
        pipeline = get_pipeline()
        result = pipeline.recommend_and_link(
            image_paths=paths,
            user_task=message,
            conversation_history=conv_history,
        )

        # helpers
        def _choices_render(choices):
            names = [c["name"] for c in choices]
            md_cards = "\n\n".join(
                f"{_fmt_toolcard(c['name'])}\n\n> **Score:** {float(c.get('accuracy',0.0)):.1f}%\n\n{c.get('why','')}"
                for c in choices
            ) if names else "—"
            table = _choices_table_md(choices)
            return names, md_cards, table

        # 4) needs clarification
        if result["conversation"]["status"] == "needs_clarification":
            q = result["conversation"]["question"]
            ctx = result["conversation"]["context"]
            opts = result["conversation"].get("options", [])
            resp = f"I need more information:\n{q}\n\n"
            if opts:
                resp += "Options:\n" + "\n".join(f"• {o}" for o in opts) + "\n\n"
            resp += f"_{ctx}_"
            chat_history[-1][1] = resp

            status_done = gr.update(value="ℹ️ More info needed.", visible=True)
            enable_inputs = (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
            yield (chat_history, history_rows, "", "", conv_history,
                   empty_radio, "—",
                   preview_acc_upd, preview_img_upd,
                   gr.update(visible=False),               # refine hidden in clarify mode
                   [],                                      # excluded_names
                   status_done,
                   *enable_inputs)
            return

        # 5) choices (up to NUM_CHOICES)
        if result.get("choices"):
            names, md_cards, md_table = _choices_render(result["choices"])
            top = result["choices"][0]
            demo = top.get("demo_link", "")

            table_block = f"\n\n**Top candidates** (up to {os.getenv('NUM_CHOICES', '3')}):\n\n" + md_table
            chat_history[-1][1] = (
                f"I recommend **{top['name']}** ({float(top.get('accuracy',0.0)):.1f}% match)\n\n"
                f"_{top.get('why','')}_" + table_block
            )

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_rows = history_rows + [[ts, message[:80], top["name"], "yes" if demo else "no"]]

            radio_update = gr.update(choices=names, value=names[0] if names else None)

            status_done = gr.update(value="✅ Ready.", visible=True)
            enable_inputs = (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
            yield (chat_history, history_rows, top["name"], demo, conv_history,
                   radio_update, md_cards,
                   preview_acc_upd, preview_img_upd,
                   gr.update(visible=True),                 # refine visible only now (complete)
                   names,                                   # excluded_names populated for refine
                   status_done,
                   *enable_inputs)
            return

        # 6) no suitable tools (terminal)
        reason = result.get("reason")
        reason_line = f"**Reason:** `{reason}`\n\n" if reason else ""
        chat_history[-1][1] = (
            "❌ No suitable tools found.\n\n"
            + reason_line
            + result.get("explanation", "")
        )

        # Clear the choices UI & hide refine
        radio_update   = gr.update(choices=[], value=None)  # clear radio
        choices_md_upd = "—"                                # clear markdown table/cards
        refine_hidden  = gr.update(visible=False)           # hide “Find alternatives”

        status_done = gr.update(value="✅ Ready.", visible=True)
        enable_inputs = (
            gr.update(interactive=True),  # message textbox
            gr.update(interactive=True),  # Send
            gr.update(interactive=True),  # Files
        )

        yield (
            chat_history,
            history_rows,
            "",                  # chosen_tool
            "",                  # demo_link
            conv_history,
            radio_update,        # choices_radio
            choices_md_upd,      # choices_md
            preview_acc_upd,     # preview accordion (unchanged)
            preview_img_upd,     # preview image (unchanged)
            refine_hidden,       # refine accordion hidden
            [],                  # excluded_names state reset
            status_done,
            *enable_inputs
        )

    return handle_message

# --- UI -----------------------------------------------------------------------
def create_interface():
    with gr.Blocks(title="Imaging Tool Finder", theme=gr.themes.Soft()) as demo:
        conversation_history = gr.State([])
        excluded_names = gr.State([])   # keep latest recommended names for refine

        gr.Markdown(
            "# 🧭 Imaging Software Finder\n"
            "Upload an image/volume/stack and describe your task. "
            "I'll recommend the best tools with scores and a demo link.\n\n"
            "_Tip: include modality (CT/MRI/microscopy), operation (segment/denoise/register), and objects of interest._"
        )

        with gr.Row():
            # LEFT
            with gr.Column(scale=7):
                files = gr.File(
                    label="Images / volumes (drag & drop multiple: DICOM zip/folder, NIfTI, TIFF, PNG/JPEG, etc.)",
                    file_types=None,
                    file_count="multiple",
                )

                # Collapsed, hidden-by-default preview
                with gr.Accordion("Preview", open=False, visible=False) as preview_acc:
                    preview_img = gr.Image(label="", interactive=False, value=None, visible=True)

                chatbot = gr.Chatbot(label="Conversation", type="tuples")

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your request",
                        placeholder="e.g., 'CT: segment lungs' or 'Microscopy TIFF: denoise & register stack'",
                        lines=2,
                    )
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")

                # Small status area (shows while working)
                status_md = gr.Markdown(visible=False)

            # RIGHT
            with gr.Column(scale=5):
                chosen_tool = gr.Markdown(label="Selected Tool")
                demo_link = gr.Textbox(
                    label="Demo link",
                    show_copy_button=True,
                    interactive=False,
                )

                with gr.Accordion("Top choices (details)", open=False):
                    choices_radio = gr.Radio(
                        label="Pick a tool",
                        choices=[],
                        interactive=True,
                    )
                    choices_md = gr.Markdown(value="—")

                # Refine UI — hidden by default; wired to closure after handler is created
                with gr.Accordion("Not a good fit? Find alternatives", open=False, visible=False) as refine_acc:
                    refine_feedback = gr.Textbox(
                        label="Why didn't this fit? (optional)",
                        placeholder="e.g., expects DICOM; my file is TIFF • wrong organ • needs GPU",
                        lines=2,
                    )
                    refine_btn = gr.Button("Find alternatives")

                history_df = gr.Dataframe(
                    headers=["Time", "Request", "Tool", "Demo"],
                    label="History",
                    row_count=5,
                    value=[],
                    type="array",
                )

        handle_message = _make_handler()

        def _refine(message, chatbot, files, history_rows, conv_history, excluded, feedback):
            # Build control tags for a refine round
            tag_refine = "[REFINE]"
            tag_excl = f"[EXCLUDE:{'|'.join(excluded)}]" if excluded else ""
            fb = f" {feedback.strip()}" if feedback and feedback.strip() else ""

            msg2 = (message or "").strip()
            control = f"{tag_refine}{tag_excl}{fb}"
            msg2 = f"{msg2}\n{control}" if msg2 else control

            gen = handle_message(msg2, chatbot, files, history_rows, conv_history)
            for step in gen:
                yield step


        # Send
        submit.click(
            handle_message,
            inputs=[msg, chatbot, files, history_df, conversation_history],
            outputs=[
                chatbot,        # streams
                history_df,     # table
                chosen_tool,
                demo_link,
                conversation_history,
                choices_radio,  # updated via gr.update(choices=..., value=...)
                choices_md,     # detailed markdown cards
                preview_acc,    # accordion visibility
                preview_img,    # preview image
                refine_acc,     # refine visibility (only on complete)
                excluded_names, # keep names for refine
                status_md,      # status banner (loading / done)
                msg,            # disable / enable while working
                submit,         # disable / enable while working
                files,          # disable / enable while working
            ],
        ).then(_clear_textbox, inputs=None, outputs=[msg])

        msg.submit(
            handle_message,
            inputs=[msg, chatbot, files, history_df, conversation_history],
            outputs=[
                chatbot, history_df, chosen_tool, demo_link, conversation_history,
                choices_radio, choices_md,
                preview_acc, preview_img,
                refine_acc,
                excluded_names,  # <-- keep names for refine
                status_md, msg, submit, files
            ],
        ).then(_clear_textbox, inputs=None, outputs=[msg])

        refine_btn.click(
            _refine,
            inputs=[msg, chatbot, files, history_df, conversation_history, excluded_names, refine_feedback],
            outputs=[
                chatbot, history_df, chosen_tool, demo_link, conversation_history,
                choices_radio, choices_md,
                preview_acc, preview_img,
                refine_acc,
                excluded_names,   # update for the next round too
                status_md, msg, submit, files,
            ],
        )

        # Clear ALL (chat, history, selections, preview, refine, status) and re-enable inputs
        clear.click(
            lambda: (
                [],  # chatbot
                "",  # chosen_tool
                "",  # demo_link
                [],  # conversation_history
                gr.update(choices=[], value=None),   # choices_radio reset
                "—",                                 # choices_md reset
                gr.update(visible=False),            # preview accordion hidden
                gr.update(value=None, visible=False),# preview img hidden
                gr.update(visible=False),            # refine accordion hidden
                [],                                  # excluded_names reset (if present in outputs)
                gr.update(value=None, visible=False),# status hidden
                gr.update(interactive=True),         # msg enabled
                gr.update(interactive=True),         # submit enabled
                gr.update(value=None, interactive=True), # files cleared & enabled
            ),
            inputs=None,
            outputs=[
                chatbot, chosen_tool, demo_link, conversation_history,
                choices_radio, choices_md,
                preview_acc, preview_img, refine_acc,
                excluded_names,
                status_md, msg, submit, files
            ],
        )


    return demo

# RUN THE APP ------------------------------------------------------------------
def _bind_host() -> str:
    if os.getenv("BIND_HOST"):
        return os.getenv("BIND_HOST")
    in_docker = os.path.exists("/.dockerenv")
    return "0.0.0.0" if in_docker else "127.0.0.1"

def launch():
    host = _bind_host()
    port = int(os.getenv("PORT", "7860"))
    ui = create_interface()

    try:
        ui.queue(api_open=False, max_size=10, default_concurrency_limit=3).launch(
            server_name=host,
            server_port=port,
            inbrowser=False,
            show_error=True,
            share=bool(os.getenv("SHARE", False)),
            favicon_path=os.path.join(ROOT, "assets", "favicon.ico"),
            auth=None if not os.getenv("GRADIO_AUTH") else lambda x, y: (
                x == os.getenv("GRADIO_USERNAME"),
                y == os.getenv("GRADIO_PASSWORD"),
            ),
        )
    except TypeError:
        # Older Gradio
        ui.queue(api_open=False, max_size=10).launch(
            server_name=host,
            server_port=port,
            inbrowser=False,
            show_error=True,
            share=bool(os.getenv("SHARE", False)),
            favicon_path=os.path.join(ROOT, "assets", "favicon.ico"),
            auth=None if not os.getenv("GRADIO_AUTH") else lambda x, y: (
                x == os.getenv("GRADIO_USERNAME"),
                y == os.getenv("GRADIO_PASSWORD"),
            ),
        )

if __name__ == "__main__":
    launch()
