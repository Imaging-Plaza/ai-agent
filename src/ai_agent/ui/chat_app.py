from __future__ import annotations

import os
import sys
import logging
from typing import List, Dict

# Ensure ai_agent is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Set up logging
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

import logging.handlers
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

log = logging.getLogger("chat_app")
log.info("Starting Chat-based Gradio UI")

import gradio as gr

from ai_agent.api.pipeline import RAGImagingPipeline
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.ui.chat_interface import respond

# ============================================================================
# Initialize pipeline and doc index
# ============================================================================

INDEX_DIR = os.getenv("RAG_INDEX_DIR", "artifacts/rag_index")

_pipe: RAGImagingPipeline | None = None
_DOCS: List[SoftwareDoc] = []
_DOC_BY_NAME: Dict[str, SoftwareDoc] = {}

def get_pipeline() -> RAGImagingPipeline:
    global _pipe, _DOCS, _DOC_BY_NAME
    if _pipe is None:
        _pipe = RAGImagingPipeline(index_dir=INDEX_DIR)
        try:
            _DOCS = list(_pipe.index.docs.values())
            _DOC_BY_NAME = {d.name: d for d in _DOCS if getattr(d, "name", None)}
            log.info("Loaded %d tools from index", len(_DOCS))
        except Exception:
            _DOCS, _DOC_BY_NAME = [], {}
            log.exception("Failed to load docs from index")
        log.info("Pipeline ready")
    return _pipe

# Initialize pipeline at startup
get_pipeline()

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


# ============================================================================
# Gradio interface
# ============================================================================

def create_chat_interface():
    """Create the chat-based Gradio interface."""
    
    with gr.Blocks(
        title="Imaging Assistant",
        theme=gr.themes.Soft(),
        fill_height=True,
    ) as demo:
        gr.Markdown(
            "# 🤖 Imaging Software Assistant\n"
            "Chat with me to find the right imaging tools! Upload files and describe your task.\n\n"
            "_I can recommend tools, run demos, and help you with medical/scientific imaging workflows._"
        )
        
        with gr.Row(equal_height=True):
            # ================================================================
            # LEFT: Chat section
            # ================================================================
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="Chat",
                    type="messages",
                    height=600,
                    show_copy_button=True,
                    avatar_images=("👤", "🤖"),
                )
                
                with gr.Row():
                    with gr.Column(scale=8):
                        msg_input = gr.Textbox(
                            label="Your message",
                            placeholder=(
                                "e.g., 'I need to segment lungs in CT scans' or "
                                "'Find tools for microscopy image denoising'"
                            ),
                            lines=2,
                        )
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="📎 Attach files",
                            file_count="multiple",
                            file_types=[
                                ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
                                ".tif", ".tiff", ".dcm", ".nii", ".nii.gz",
                                ".csv", ".json", ".xml",
                                ".mp3", ".wav", ".mp4", ".avi",
                            ],
                        )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear chat", scale=1)
            
            # ================================================================
            # RIGHT: Dev / Conversation State section
            # ================================================================
            with gr.Column(scale=3, visible=True):
                # Use the JSON label as the title to avoid extra vertical gap
                state_display = gr.JSON(
                    label="🔧 Conversation State",
                    value={},
                    show_label=True,
                    height=600,
                    max_height=1000,
                )
        
        # Hidden state
        chat_state = gr.State({})
        
        # ====================================================================
        # Chat handler
        # ====================================================================
        
        def handle_chat(message: str, history: List[dict], files: List, state_dict: dict):
            """
            Handle chat message with streaming response.
            Yields updated history and state after each step.
            """
            # Convert Gradio messages format to internal format if needed
            if not history:
                history = []
            
            # Show user message immediately
            user_msg = {"role": "user", "content": message or ""}
            
            # Add file attachments to user message
            if files:
                file_list = "\n".join(
                    [
                        f"📎 {os.path.basename(f.name if hasattr(f, 'name') else str(f))}"
                        for f in files
                    ]
                )
                if message:
                    user_msg["content"] = f"{message}\n\n{file_list}"
                else:
                    user_msg["content"] = file_list
            
            history.append(user_msg)
            yield history, state_dict, gr.update()
            
            # If files were uploaded, build and show preview immediately
            if files:
                file_paths = []
                for f in files:
                    if isinstance(f, str):
                        file_paths.append(f)
                    elif hasattr(f, "name"):
                        file_paths.append(f.name)
                
                if file_paths:
                    # Build preview
                    from ai_agent.utils.previews import _build_preview_for_vlm
                    try:
                        preview_path, meta_text = _build_preview_for_vlm(file_paths)
                        if preview_path:
                            # Show preview message
                            preview_text = "📋 **Preview for analysis:**"
                            if meta_text:
                                preview_text += f"\n\n_{meta_text}_"
                            history.append(
                                {"role": "assistant", "content": preview_text}
                            )
                            history.append(
                                {
                                    "role": "assistant",
                                    "content": {"path": preview_path},
                                }
                            )
                            yield history, state_dict, gr.update()
                    except Exception as e:
                        log.warning("Preview generation failed: %r", e)
            
            # Show "thinking" indicator for agent processing
            thinking_msg = {"role": "assistant", "content": "🤔 Finding tools..."}
            history.append(thinking_msg)
            yield history, state_dict, gr.update()
            
            # Call respond function
            try:
                reply, new_state = respond(
                    message=message or "",
                    files=files or [],
                    state_dict=state_dict,
                    doc_index=_DOC_BY_NAME,
                )
                
                # Remove thinking indicator
                history.pop()
                
                # Add assistant response with rich media
                # Build text content first
                text_content = reply.text
                
                # Add file links
                if reply.files:
                    text_content += "\n\n" + "\n".join(
                        [f"📎 [{label}]({path})" for path, label in reply.files]
                    )
                
                # Add JSON
                if reply.json_data:
                    import json
                    text_content += (
                        "\n\n```json\n"
                        + json.dumps(reply.json_data, indent=2)
                        + "\n```"
                    )
                
                # Add code blocks
                for lang, code in reply.code_blocks:
                    text_content += f"\n\n```{lang}\n{code}\n```"
                
                # Add text message first
                history.append({"role": "assistant", "content": text_content})
                
                # Add each image as a separate message for proper Gradio rendering
                for img_path in reply.images:
                    if os.path.exists(img_path):
                        history.append(
                            {"role": "assistant", "content": {"path": img_path}}
                        )
                
                # Update state displays
                state_dict_updated = new_state.to_dict()
                
                yield (
                    history,
                    state_dict_updated,
                    gr.update(value=state_dict_updated),
                )
            
            except Exception as e:
                log.exception("Error in chat handler")
                history.pop()  # Remove thinking indicator
                error_msg = {
                    "role": "assistant",
                    "content": (
                        f"❌ Error: {str(e)}\n\n"
                        "Please try again or rephrase your request."
                    ),
                }
                history.append(error_msg)
                yield history, state_dict, gr.update()
        
        # ====================================================================
        # Event handlers
        # ====================================================================
        
        submit_btn.click(
            handle_chat,
            inputs=[msg_input, chatbot, file_input, chat_state],
            outputs=[chatbot, chat_state, state_display],
        ).then(
            lambda: ("", None),  # Clear inputs
            inputs=None,
            outputs=[msg_input, file_input],
        )
        
        msg_input.submit(
            handle_chat,
            inputs=[msg_input, chatbot, file_input, chat_state],
            outputs=[chatbot, chat_state, state_display],
        ).then(
            lambda: ("", None),  # Clear inputs
            inputs=None,
            outputs=[msg_input, file_input],
        )
        
        def clear_chat():
            """Reset everything."""
            return [], {}, gr.update(value={})
        
        clear_btn.click(
            clear_chat,
            inputs=None,
            outputs=[chatbot, chat_state, state_display],
        )
    
    return demo

# ============================================================================
# Launch
# ============================================================================

def _bind_host() -> str:
    if os.getenv("BIND_HOST"):
        return os.getenv("BIND_HOST")
    in_docker = os.path.exists("/.dockerenv")
    return "0.0.0.0" if in_docker else "127.0.0.1"


def launch():
    """Launch the chat interface."""
    host = _bind_host()
    preferred = int(os.getenv("PORT", "7860"))
    max_tries = int(os.getenv("PORT_TRIES", "10"))
    allow_fallback = str(os.getenv("ALLOW_PORT_FALLBACK", "1")).lower() in ("1", "true", "yes", "on")
    
    ui = create_chat_interface()
    
    last_err = None
    for attempt in range(max_tries if allow_fallback else 1):
        port = preferred + attempt
        try:
            ui.queue(max_size=10).launch(
                server_name=host,
                server_port=port,
                inbrowser=False,
                show_error=True,
                share=bool(os.getenv("SHARE", False)),
            )
            if attempt > 0:
                log.info("Launched on fallback port %d (preferred %d was busy)", port, preferred)
            return
        except OSError as e:
            last_err = e
            busy = "Cannot find empty port" in str(e)
            if not busy or attempt == (max_tries - 1) or not allow_fallback:
                raise
            log.warning("Port %d busy; trying %d", port, port + 1)
    
    if last_err:
        raise last_err


if __name__ == "__main__":
    launch()
