import logging
import os
import json
from typing import List, Dict

import gradio as gr

from ai_agent.utils.previews import _build_preview_for_vlm
from ai_agent.retriever.software_doc import SoftwareDoc

from .handlers import respond

log = logging.getLogger("chat_components")


def create_chat_interface(doc_index: Dict[str, SoftwareDoc]):
    """
    Create the chat-based Gradio interface.
    
    Args:
        doc_index: Mapping of tool name -> SoftwareDoc for formatting
    
    Returns:
        Gradio Blocks interface
    """
    
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
                state_display = gr.JSON(
                    label="🔧 Conversation State",
                    value={},
                    show_label=True,
                    height=600,
                    max_height=1000,
                )
        
        chat_state = gr.State({})
        
        # ====================================================================
        # Event Handlers
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
                    doc_index=doc_index,
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
        
        def clear_chat():
            """Reset everything."""
            return [], {}, gr.update(value={})
        
        # Wire up events
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
        
        clear_btn.click(
            clear_chat,
            inputs=None,
            outputs=[chatbot, chat_state, state_display],
        )
    
    return demo
