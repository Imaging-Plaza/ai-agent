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

import gradio as gr
from pandas import DataFrame

from retriever.software_doc import SoftwareDoc
from api.pipeline import RAGImagingPipeline
from agent.agent import run_agent
from agent.tools.gradio_space_tool import tool_run_example, RunExampleInput

from utils.file_validator import FileValidator
from utils.tags import strip_tags, parse_exclusions
from utils.previews import _build_preview_for_vlm
from utils.image_analyzer import _to_supported_png_dataurl

# --- config -------------------------------------------------------------------
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/rag_index")

# --- pipeline singleton + in-memory doc index --------------------------------
_pipe: Optional[RAGImagingPipeline] = None
_DOCS: List[SoftwareDoc] = []
_DOC_BY_NAME: Dict[str, SoftwareDoc] = {}

def get_pipeline() -> RAGImagingPipeline:
    global _pipe, _DOCS, _DOC_BY_NAME
    if _pipe is None:
        _pipe = RAGImagingPipeline(index_dir=INDEX_DIR)
        try:
            _DOCS = list(_pipe.index.docs.values())
            _DOC_BY_NAME = {d.name: d for d in _DOCS if getattr(d, "name", None)}
            log.info("Loaded %d tools from FAISS meta in %s", len(_DOCS), INDEX_DIR)
        except Exception:
            _DOCS, _DOC_BY_NAME = [], {}
            log.exception("Failed to load docs from FAISS sidecar")
        log.info("Pipeline ready")
    return _pipe

def refresh_ui_docs_from_index():
    global _pipe, _DOCS, _DOC_BY_NAME
    if _pipe is None:
        return
    try:
        _DOCS = list(_pipe.index.docs.values())
        _DOC_BY_NAME = {d.name: d for d in _DOCS if getattr(d, "name", None)}
        log.info("UI docs refreshed from FAISS: %d tools", len(_DOCS))
    except Exception:
        _DOCS, _DOC_BY_NAME = [], {}
        log.exception("Failed to refresh UI docs from FAISS")

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

# --- Chat format helpers (pairs <-> messages) --------------------------------
def _msgs_to_pairs(msgs):
    """Convert Chatbot(messages) -> legacy list[[user, assistant]]."""
    if not msgs:
        return []
    # If already pairs, pass through
    if isinstance(msgs[0], (list, tuple)):
        return msgs
    pairs = []
    pending_user = None
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else None
        content = (m.get("content", "") if isinstance(m, dict) else "")
        if role == "user":
            if pending_user is not None:
                pairs.append([pending_user, None])
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                pairs.append(["", content])
            else:
                pairs.append([pending_user, content])
                pending_user = None
    if pending_user is not None:
        pairs.append([pending_user, None])
    return pairs

def _pairs_to_msgs(pairs):
    """Convert legacy list[[user, assistant]] -> Chatbot(messages)."""
    if not pairs:
        return []
    # If already messages, pass through
    if isinstance(pairs[0], dict):
        return pairs
    msgs = []
    for item in pairs:
        user, assistant = (item + [None, None])[:2] if isinstance(item, list) else (None, None)
        if user is not None:
            msgs.append({"role": "user", "content": user})
        if assistant is not None:
            msgs.append({"role": "assistant", "content": assistant})
    return msgs

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

# --- Radio selection handler --------------------------------------------
def _on_pick_tool(selected_name: str, choices_map: dict):
    """
    Update the right panel when a radio option is picked.
    choices_map: {tool_name: {accuracy, why, demo_link, ...}}
    """
    c = (choices_map or {}).get(selected_name, {})
    if not selected_name or not c:
        return gr.update(value=""), gr.update(value="")
    md = (
        _fmt_toolcard(selected_name)
        + f"\n\n> **Score:** {float(c.get('accuracy', 0.0)):.1f}%\n\n"
        + (c.get("why", "") or "")
    )
    demo = c.get("demo_link", "") or ""
    return gr.update(value=md), gr.update(value=demo)

# --- Chat handler (streaming + status/disable) --------------------------------
def _make_handler():
    def handle_message(message: str,
                    chat_history,
                    files,
                    history_rows,
                    conv_history,
                    banlist_state):
        # normalize inputs
        if isinstance(history_rows, DataFrame):
            history_rows = history_rows.values.tolist()
        history_rows = history_rows or []
        # Accept both old pair format and new messages format from Chatbot
        if chat_history and isinstance(chat_history, list) and isinstance(chat_history[0], dict):
            chat_pairs = _msgs_to_pairs(chat_history)
        else:
            chat_pairs = chat_history or []
        conv_history = conv_history or []
        banlist = set(banlist_state or set())

        empty_radio = gr.update(choices=[], value=None)
        status_idle = gr.update(value=None, visible=False)
        choices_acc_hidden = gr.update(visible=False, open=False)
        empty_choices_map = {}

        # Demo controls hidden by default
        demo_link_hidden = gr.update(value="", visible=False)
        run_demo_hidden = gr.update(visible=False)

        # Check if user provided any input
        raw_message = (message or "")
        has_any_text = bool(raw_message.strip())
        has_files    = bool(files)

        if (not has_any_text) and (not has_files):
            # nothing to do → DO NOT disable inputs; just return quietly
            yield (_pairs_to_msgs(chat_pairs), history_rows, "", demo_link_hidden, run_demo_hidden, conv_history,
                empty_radio, "—",
                gr.update(visible=False),              # preview accordion hidden
                gr.update(value=None, visible=False),  # preview image hidden
                status_idle,                           # status hidden
                gr.update(interactive=True),           # msg enabled
                gr.update(interactive=True),           # submit enabled
                gr.update(interactive=True),           # files enabled
                gr.update(interactive=True),           # clear enabled
                banlist,
                choices_acc_hidden,                    # choices accordion hidden
                empty_choices_map)                     # last choices map reset
            return

        # Clean message (remove control tags for display)
        visible_msg = strip_tags(raw_message)
        if not visible_msg:
            visible_msg = "Find tools"

        # Parse any explicit exclusions from tags like [EXCLUDE:tool1|tool2]
        new_exclusions = set(parse_exclusions(raw_message))
        banlist |= new_exclusions

        # Disable inputs while processing
        disable_inputs = (
            gr.update(interactive=False),  # msg
            gr.update(interactive=False),  # submit
            gr.update(interactive=False),  # files
            gr.update(interactive=False),  # clear button
        )

        chat_pairs = chat_pairs + [[visible_msg, None]]
        conv_history = conv_history + [f"User: {visible_msg}"]
        status = gr.update(value="🔄 Validating files…", visible=True)
        yield (_pairs_to_msgs(chat_pairs), history_rows, "", demo_link_hidden, run_demo_hidden, conv_history,
            empty_radio, "—",
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            status,
            *disable_inputs,
            banlist,
            choices_acc_hidden,                     
            empty_choices_map)

        # 1) file validation
        paths = _coerce_gradio_files_to_paths(files)
        ok, why_not = _validate_files(paths)
        if not ok:
            chat_pairs[-1][1] = f"⚠️ File issues detected.\n\n{why_not}"
            status_done = gr.update(value="⚠️ File validation failed.", visible=True)
            # re-enable inputs
            enable_inputs = (
                gr.update(interactive=True),  # msg
                gr.update(interactive=True),  # submit
                gr.update(interactive=True),  # files
                gr.update(interactive=True),  # clear button
            )
            yield (_pairs_to_msgs(chat_pairs), history_rows, "", demo_link_hidden, run_demo_hidden, conv_history,
                empty_radio, "—",
                gr.update(visible=False),
                gr.update(value=None, visible=False),
                status_done,
                *enable_inputs,
                banlist,
                choices_acc_hidden,                     
                empty_choices_map)                      
            return

        # 2) build preview (collapsed & only visible if present)
        status = gr.update(value="🖼️ Building preview…", visible=True)
        yield (_pairs_to_msgs(chat_pairs), history_rows, "", demo_link_hidden, run_demo_hidden, conv_history,
            empty_radio, "—",
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            status,
            *disable_inputs,
            banlist,
            choices_acc_hidden,                     
            empty_choices_map)

        preview_path = None
        meta_text = None
        try:
            preview_path, meta_text = _build_preview_for_vlm(paths)
        except Exception:
            preview_path = None

        preview_acc_upd = gr.update(visible=bool(preview_path))
        preview_img_upd = gr.update(value=preview_path, visible=bool(preview_path))

        yield (_pairs_to_msgs(chat_pairs), history_rows, "", demo_link_hidden, run_demo_hidden, conv_history,
            empty_radio, "—",
            preview_acc_upd,
            preview_img_upd,
            gr.update(value="📚 Running agent…", visible=True),
            *disable_inputs,
            banlist,
            choices_acc_hidden,                         
            empty_choices_map)

        # 3) run agent with conversation history
        log.info("Running agent for task: %s", visible_msg)
        data_url = None
        if preview_path:
            try:
                data_url = _to_supported_png_dataurl(preview_path)
            except Exception:
                data_url = None
        
        # Extract original file formats
        original_formats = []
        if paths:
            for pth in paths:
                ext = os.path.splitext(pth)[1].lower().lstrip('.')
                if ext == 'gz' and pth.lower().endswith('.nii.gz'):
                    ext = 'nii.gz'
                if ext and ext not in original_formats:
                    original_formats.append(ext)
        
        agent_sel = run_agent(
            visible_msg, 
            image_data_url=data_url, 
            excluded=list(banlist), 
            original_formats=original_formats, 
            image_meta=meta_text, 
            conversation_history=conv_history
        )
        result = agent_sel.to_legacy_dict()
        log.info("Agent tool calls: %s", result["tool_calls"])

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
            chat_pairs[-1][1] = resp
            
            # Update conversation history
            conv_history = conv_history + [f"Assistant: {resp}"]

            status_done = gr.update(value="ℹ️ More info needed.", visible=True)
            enable_inputs = (
                gr.update(interactive=True),  # msg
                gr.update(interactive=True),  # submit
                gr.update(interactive=True),  # files
                gr.update(interactive=True),  # clear button
            )
            yield (_pairs_to_msgs(chat_pairs), history_rows, "", demo_link_hidden, run_demo_hidden, conv_history,
                empty_radio, "—",
                preview_acc_upd, preview_img_upd,
                status_done,
                *enable_inputs,
                banlist,
                choices_acc_hidden,                       
                empty_choices_map)                        
            return

        # 5) choices (up to NUM_CHOICES)
        if result.get("choices"):
            names, md_cards, md_table = _choices_render(result["choices"])
            top = result["choices"][0]
            demo = top.get("demo_link", "")

            table_block = f"\n\n**Top candidates** (up to {os.getenv('NUM_CHOICES', '3')}):\n\n" + md_table
            resp = (
                f"I recommend **{top['name']}** ({float(top.get('accuracy',0.0)):.1f}% match)\n\n"
                f"_{top.get('why','')}_" + table_block
            )
            chat_pairs[-1][1] = resp
            
            # Update conversation history
            conv_history = conv_history + [f"Assistant: {resp}"]

            # Only log to history table if this is a primary request (files provided OR first request)
            # Follow-up requests like "another tool" don't create new history rows
            is_primary_request = bool(paths) or len(banlist) == len(result["choices"])
            if is_primary_request:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history_rows = history_rows + [[ts, visible_msg[:80], top["name"], "yes" if demo else "no"]]

            radio_update = gr.update(choices=names, value=names[0] if names else None)
            choices_map = {c["name"]: c for c in result["choices"]}
            choices_acc_upd = gr.update(visible=True, open=True)

            # Add ALL recommended tools to banlist so follow-up "another tool" requests exclude them
            for choice in result["choices"]:
                if choice.get("name"):
                    banlist.add(choice["name"])

            status_done = gr.update(value="✅ Ready.", visible=True)
            # Show demo controls once a tool is found
            demo_link_upd = gr.update(value=demo, visible=True)
            run_demo_upd = gr.update(visible=True)
            enable_inputs = (
                gr.update(interactive=True),  # msg
                gr.update(interactive=True),  # submit
                gr.update(interactive=True),  # files
                gr.update(interactive=True),  # clear button
            )
            yield (_pairs_to_msgs(chat_pairs), history_rows, top["name"], demo_link_upd, run_demo_upd, conv_history,
                radio_update, md_cards,
                preview_acc_upd, preview_img_upd,
                status_done,
                *enable_inputs,
                banlist,
                choices_acc_upd,                         # show accordion
                choices_map)                             # keep choices mapping
            return

        # 6) no suitable tools (terminal)
        reason = result.get("reason")
        reason_line = f"**Reason:** `{reason}`\n\n" if reason else ""
        resp = (
            "❌ No suitable tools found.\n\n"
            + reason_line
            + result.get("explanation", "")
        )
        chat_pairs[-1][1] = resp
        
        # Update conversation history
        conv_history = conv_history + [f"Assistant: {resp}"]

        # Clear the choices UI 
        radio_update   = gr.update(choices=[], value=None)  # clear radio
        choices_md_upd = "—"                                # clear markdown table/cards

        status_done = gr.update(value="✅ Ready.", visible=True)
        enable_inputs = (
            gr.update(interactive=True),  # message textbox
            gr.update(interactive=True),  # Send
            gr.update(interactive=True),  # Files
            gr.update(interactive=True),  # Clear button
        )

        yield (
            _pairs_to_msgs(chat_pairs),
            history_rows,
            "",                  # chosen_tool
            demo_link_hidden,     # demo_link hidden
            run_demo_hidden,      # run_demo_btn hidden
            conv_history,
            radio_update,        # choices_radio
            choices_md_upd,      # choices_md
            preview_acc_upd,     # preview accordion (unchanged)
            preview_img_upd,     # preview image (unchanged)
            status_done,
            *enable_inputs,
            banlist,
            choices_acc_hidden,  # hide accordion
            empty_choices_map    # reset choices mapping
        )
    
    return handle_message


# --- UI -----------------------------------------------------------------------
def create_interface():
    with gr.Blocks(title="Imaging Tool Finder", theme=gr.themes.Soft()) as demo:
        conversation_history = gr.State([])
        banlist_state = gr.State(set())          # persistent set of banned tool names
        last_choices_state = gr.State({})        

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
                    preview_img = gr.Image(label="", interactive=False, value=None, visible=True, type="filepath")
                with gr.Accordion("Demo result", open=False, visible=False) as demo_result_acc:
                    demo_result_img = gr.Image(label="", interactive=False, value=None, visible=True)
                    demo_result_file = gr.File(label="Download result file", visible=False)

                chatbot = gr.Chatbot(label="Conversation", type="messages")

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
                # Hide demo controls until a tool is found
                demo_link = gr.Textbox(
                    label="Demo link",
                    show_copy_button=True,
                    interactive=False,
                    visible=False,
                )
                run_demo_btn = gr.Button("Run demo on preview", variant="secondary", visible=False)

                with gr.Accordion("Top choices (details)", open=False, visible=False) as choices_acc:
                    choices_radio = gr.Radio(
                        label="Pick a tool",
                        choices=[],
                        interactive=True,
                    )
                    choices_md = gr.Markdown(value="—")

                history_df = gr.Dataframe(
                    headers=["Time", "Request", "Tool", "Demo"],
                    label="History",
                    row_count=5,
                    value=[],
                    type="array",
                )

        handle_message = _make_handler()

        # Send
        submit.click(
            handle_message,
            inputs=[msg, chatbot, files, history_df, conversation_history, banlist_state],
            outputs=[
                chatbot,        # streams
                history_df,     # table
                chosen_tool,
                demo_link,
                run_demo_btn,
                conversation_history,
                choices_radio,  # updated via gr.update(choices=..., value=...)
                choices_md,     # detailed markdown cards
                preview_acc,    # accordion visibility
                preview_img,    # preview image
                status_md,      # status banner (loading / done)
                msg,            # disable / enable while working
                submit,         # disable / enable while working
                files,          # disable / enable while working
                clear,          # disable / enable while working
                banlist_state,
                choices_acc,          # show/hide top choices accordion
                last_choices_state,   # keep latest choices mapping
            ],
        ).then(_clear_textbox, inputs=None, outputs=[msg])


        msg.submit(
            handle_message,
            inputs=[msg, chatbot, files, history_df, conversation_history, banlist_state],
            outputs=[
                chatbot, history_df, chosen_tool, demo_link, run_demo_btn, conversation_history,
                choices_radio, choices_md,
                preview_acc, preview_img,
                status_md, msg, submit, files, clear,
                banlist_state,
                choices_acc,
                last_choices_state,
            ],
        ).then(_clear_textbox, inputs=None, outputs=[msg])

        def _run_selected_demo(selected_name: str, demo_url: str, uploaded_files):
            # Use the original uploaded file(s) instead of preview; prefer .tif/.tiff if present
            if not selected_name:
                return gr.update(visible=False), gr.update(visible=False, value=None), gr.update(value=None, visible=False), gr.update(value="⚠️ Select a tool first.", visible=True)
            paths = _coerce_gradio_files_to_paths(uploaded_files)
            if not paths:
                return gr.update(visible=False), gr.update(visible=False, value=None), gr.update(value=None, visible=False), gr.update(value="⚠️ Please upload an image first.", visible=True)
            # prefer tif/tiff if available (per remote app constraints), else first file
            pick = None
            for p in paths:
                ext = os.path.splitext(p)[1].lower()
                if ext in (".tif", ".tiff"):
                    pick = p
                    break   
            if not pick:
                pick = paths[0]
            try:
                log.info("Run demo: tool=%s, path=%s, url=%s", selected_name, pick, demo_url)
                out = tool_run_example(RunExampleInput(tool_name=selected_name, image_path=pick, endpoint_url=demo_url or None))
                if out.ran and (out.result_preview or out.result_image):
                    preview = out.result_preview or out.result_image
                    file_upd = gr.update(value=out.result_origin, visible=bool(out.result_origin))
                    return gr.update(visible=True, open=True), gr.update(value=preview, visible=True), file_upd, gr.update(value="✅ Demo ran.", visible=True)
                note = out.notes or ""
                txt = f"ℹ️ Ran, no image returned. {note}" if out.ran else f"❌ Failed. {note}"
                return gr.update(visible=False), gr.update(value=None, visible=False), gr.update(visible=False, value=None), gr.update(value=txt, visible=True)
            except Exception as e:
                return gr.update(visible=False), gr.update(value=None, visible=False), gr.update(visible=False, value=None), gr.update(value=f"❌ Error: {e}", visible=True)

        run_demo_btn.click(
            _run_selected_demo,
            inputs=[choices_radio, demo_link, files],
            outputs=[demo_result_acc, demo_result_img, demo_result_file, status_md],
        )

        choices_radio.change(
            _on_pick_tool,
            inputs=[choices_radio, last_choices_state],
            outputs=[chosen_tool, demo_link],
        )

        # Clear ALL (chat, history, selections, preview, status) and re-enable inputs
        def clear_all():
            # Return reset state
            return (
                [],  # chatbot
                "",  # chosen_tool
                gr.update(value="", visible=False),  # demo_link hidden
                gr.update(visible=False),             # run_demo_btn hidden
                [],  # conversation_history
                gr.update(choices=[], value=None),   # choices_radio reset
                "—",                                 # choices_md reset
                gr.update(visible=False),            # preview accordion hidden
                gr.update(value=None, visible=False),# preview img hidden
                gr.update(visible=False),            # demo result accordion hidden
                gr.update(value=None, visible=False),# demo result img hidden
                gr.update(value=None, visible=False),# demo result file hidden
                gr.update(value=None, visible=False),# status hidden
                gr.update(interactive=True),         # msg enabled
                gr.update(interactive=True),         # submit enabled
                gr.update(value=None, interactive=True), # files cleared & enabled
                gr.update(interactive=True),         # clear button enabled
                set(),                               # banlist_state reset
                gr.update(visible=False, open=False),# hide choices accordion
                {},                                  # clear choices mapping
            )
        
        clear.click(
            clear_all,
            inputs=None,
            outputs=[
                chatbot, chosen_tool, demo_link, run_demo_btn, conversation_history,
                choices_radio, choices_md,
                preview_acc, preview_img,
                demo_result_acc, demo_result_img, demo_result_file,
                status_md, msg, submit, files, clear,
                banlist_state,
                choices_acc, last_choices_state,
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
    preferred = int(os.getenv("PORT", "7860"))
    max_tries = int(os.getenv("PORT_TRIES", "10"))  # try sequential ports if busy
    allow_fallback = str(os.getenv("ALLOW_PORT_FALLBACK", "1")).lower() in ("1", "true", "yes", "on")
    ui = create_interface()

    last_err = None
    for attempt in range(max_tries if allow_fallback else 1):
        port = preferred + attempt
        try:
            try:
                ui.queue(api_open=False, max_size=10, default_concurrency_limit=3).launch(
                    server_name=host,
                    server_port=port,
                    inbrowser=False,
                    show_error=True,
                    share=bool(os.getenv("SHARE", False)),
                    auth=None if not os.getenv("GRADIO_AUTH") else lambda x, y: (
                        x == os.getenv("GRADIO_USERNAME"),
                        y == os.getenv("GRADIO_PASSWORD"),
                    ),
                )
            except TypeError:
                # Older Gradio signature
                ui.queue(api_open=False, max_size=10).launch(
                    server_name=host,
                    server_port=port,
                    inbrowser=False,
                    show_error=True,
                    share=bool(os.getenv("SHARE", False)),
                    auth=None if not os.getenv("GRADIO_AUTH") else lambda x, y: (
                        x == os.getenv("GRADIO_USERNAME"),
                        y == os.getenv("GRADIO_PASSWORD"),
                    ),
                )
            if attempt > 0:
                log.info("Launched on fallback port %d (preferred %d was busy)", port, preferred)
            return
        except OSError as e:  # port busy
            last_err = e
            busy = "Cannot find empty port" in str(e)
            if not busy or attempt == (max_tries - 1) or not allow_fallback:
                raise
            log.warning("Port %d busy; trying %d", port, port + 1)

    if last_err:  # Should not reach if we raised above, but defensive
        raise last_err

if __name__ == "__main__":
    launch()