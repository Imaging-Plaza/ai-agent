import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple

from ai_agent.agent.agent import run_agent
from ai_agent.agent.tools.gradio_space_tool import tool_run_example, RunExampleInput
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.utils.file_validator import FileValidator
from ai_agent.utils.tags import strip_tags, parse_exclusions
from ai_agent.utils.previews import _build_preview_for_vlm
from ai_agent.utils.image_analyzer import _to_supported_png_dataurl
from ai_agent.utils.utils import _coerce_files_to_paths, _is_affirmative

from .state import ChatState, ChatMessage
from .formatters import format_tool_card

log = logging.getLogger("chat_handlers")


def respond(
    message: str,
    files: List[Any],
    state_dict: dict,
    doc_index: Dict[str, SoftwareDoc],
    model: str = None,
    top_k: int = None,
    num_choices: int = None,
) -> Tuple[ChatMessage, ChatState]:
    """
    Main agent response function.
    
    Args:
        message: User's text message
        files: List of uploaded files (paths or file objects)
        state_dict: Serialized ChatState
        doc_index: Mapping of tool name -> SoftwareDoc
        model: Model display name (e.g., 'gpt-4o' or 'openai/gpt-oss-120b [EPFL]')
        top_k: Number of candidates to retrieve (optional)
        num_choices: Number of tools to recommend (optional)
    
    Returns:
        (ChatMessage with reply + media, updated ChatState)
    """
    state = ChatState.from_dict(state_dict)
    reply = ChatMessage()
    
    # Coerce files to paths
    file_paths = _coerce_files_to_paths(files)
    
    # Check for empty input
    if not message.strip() and not file_paths:
        reply.text = "Please provide a message or upload files."
        return reply, state
    
    # Parse message for control tags
    clean_message = strip_tags(message)
    new_exclusions = set(parse_exclusions(message))
    state.banlist |= new_exclusions
    
    # Add user message to history
    state.conversation_history.append(f"User: {clean_message}")
    
    # ========================================================================
    # Check for demo confirmation
    # ========================================================================
    if state.pending_demo_tool and _is_affirmative(message):
        log.info("User confirmed demo run for %s", state.pending_demo_tool)
        reply.text = f"🚀 Running demo for **{state.pending_demo_tool}**...\n\n"
        
        # Use last uploaded files
        demo_files = state.last_files if state.last_files else file_paths
        if not demo_files:
            reply.text += "⚠️ No files available. Please upload an image first."
            state.pending_demo_tool = None
            state.pending_demo_url = None
            return reply, state
        
        # Prefer TIFF if available
        pick = None
        for p in demo_files:
            ext = os.path.splitext(p)[1].lower()
            if ext in (".tif", ".tiff"):
                pick = p
                break
        if not pick:
            pick = demo_files[0]
        
        try:
            demo_result = tool_run_example(RunExampleInput(
                tool_name=state.pending_demo_tool,
                image_path=pick,
                endpoint_url=state.pending_demo_url or None,
            ))
            
            if demo_result.ran and (demo_result.result_preview or demo_result.result_image):
                preview_path = demo_result.result_preview or demo_result.result_image
                reply.text += f"✅ Demo completed!\n\n"
                reply.images.append(preview_path)
                
                if demo_result.result_origin:
                    reply.files.append((demo_result.result_origin, "Download result"))
            else:
                note = demo_result.notes or "No output image returned"
                reply.text += f"ℹ️ Demo ran but {note}"
            
            state.tool_calls.append({
                "tool": "run_example",
                "tool_name": state.pending_demo_tool,
                "ran": demo_result.ran,
                "endpoint_url": demo_result.endpoint_url,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            log.exception("Demo execution failed")
            reply.text += f"❌ Error: {e}"
        
        # Clear pending demo
        state.pending_demo_tool = None
        state.pending_demo_url = None
        state.conversation_history.append(f"Assistant: {reply.text}")
        return reply, state
    
    # If user declined demo or said something else, clear pending
    if state.pending_demo_tool:
        state.pending_demo_tool = None
        state.pending_demo_url = None
    
    # ========================================================================
    # Validate files
    # ========================================================================
    if file_paths:
        try:
            valid_paths, errors = FileValidator.validate_files(file_paths)
            if errors:
                if isinstance(errors, (list, tuple)):
                    issues = "\n".join(f"• {x}" for x in errors)
                else:
                    issues = str(errors)
                reply.text = f"⚠️ File validation issues:\n\n{issues}"
                state.conversation_history.append(f"Assistant: {reply.text}")
                return reply, state
        except Exception as e:
            log.debug("FileValidator error: %r", e)
    
    # ========================================================================
    # Build preview for files
    # ========================================================================
    if file_paths:
        state.last_files = file_paths
        
        # Build VLM preview (but don't add to reply text yet)
        try:
            preview_path, meta_text = _build_preview_for_vlm(file_paths)
            state.last_preview_path = preview_path
            state.last_image_meta = meta_text
        except Exception as e:
            log.warning("Preview build failed: %r", e)
            state.last_preview_path = None
            state.last_image_meta = None
    
    # ========================================================================
    # Run agent
    # ========================================================================
    reply.text += f"🤔 Finding tools for: _{clean_message}_\n\n"
    
    data_url = None
    if state.last_preview_path:
        try:
            data_url = _to_supported_png_dataurl(state.last_preview_path)
        except Exception as e:
            log.debug(
                "Failed to build PNG data URL from preview %r: %r",
                state.last_preview_path,
                e,
            )
    
    # Extract original formats
    original_formats = []
    if file_paths:
        for pth in file_paths:
            ext = os.path.splitext(pth)[1].lower().lstrip('.')
            if ext == 'gz' and pth.lower().endswith('.nii.gz'):
                ext = 'nii.gz'
            if ext and ext not in original_formats:
                original_formats.append(ext)
    
    log.info("Running agent: task=%s, formats=%s, excluded=%s", 
             clean_message, original_formats, list(state.banlist))
    
    # Parse model configuration if provided
    model_name = None
    base_url_override = None  # Use different variable name
    if model:
        # Import here to avoid circular dependency
        from ai_agent.ui.components import get_model_config
        model_config = get_model_config(model)
        model_name = model_config.get("name")
        base_url_override = model_config.get("base_url")  # Can be None for OpenAI
        log.info(f"Model config: {model} -> name={model_name}, base_url={base_url_override}")
    
    try:
        agent_result = run_agent(
            clean_message,
            image_data_url=data_url,
            excluded=list(state.banlist),
            original_formats=original_formats,
            image_meta=state.last_image_meta,
            conversation_history=state.conversation_history,
            model=model_name,
            base_url=base_url_override if model else None,  # Only override if model selected
            top_k=top_k,
            num_choices=num_choices,
        )
    except ValueError as e:
        # Configuration error (missing API key, etc.)
        error_msg = str(e)
        log.error(f"Configuration error: {error_msg}")
        reply.text = f"⚠️ **Configuration Error**\n\n{error_msg}\n\n"
        if "EPFL_API_KEY" in error_msg:
            reply.text += "💡 **Tip:** EPFL models require VPN connection and EPFL_API_KEY in your .env file.\n\n"
            reply.text += "Try selecting an OpenAI model (gpt-4o-mini, gpt-4o) instead."
        elif "OPENAI_API_KEY" in error_msg:
            reply.text += "💡 **Tip:** Set OPENAI_API_KEY in your .env file to use OpenAI models."
        state.conversation_history.append(f"Assistant: {reply.text}")
        return reply, state
    except Exception as e:
        # Other errors (connection, API, etc.)
        error_msg = str(e)
        log.error(f"Agent execution error: {error_msg}", exc_info=True)
        reply.text = f"❌ **Error**\n\n{error_msg}\n\n"
        
        # Provide helpful hints based on error type
        if "key_model_access_denied" in error_msg or "key not allowed" in error_msg.lower():
            reply.text += "💡 **Tip:** This API key doesn't have access to this model.\n\n"
            if "gpt-4o" in error_msg or "gpt-3.5" in error_msg:
                reply.text += "If using EPFL config, try selecting an EPFL model from the dropdown (e.g., 'openai/gpt-oss-120b [EPFL]')."
            else:
                reply.text += "If using OpenAI models, make sure you're using the correct API key."
        elif "ConnectError" in error_msg or "Connection" in error_msg:
            reply.text += "💡 **Tip:** Connection failed. If using EPFL models, ensure you're connected to EPFL VPN."
        
        state.conversation_history.append(f"Assistant: {reply.text}")
        return reply, state
    
    result_dict = agent_result.to_legacy_dict()
    
    # Record tool calls
    if "tool_calls" in result_dict:
        state.tool_calls.extend(result_dict["tool_calls"])
        reply.tool_traces = result_dict["tool_calls"]
    
    # ========================================================================
    # Handle agent response
    # ========================================================================
    status = result_dict["conversation"]["status"]
    
    if status == "needs_clarification":
        # Agent needs more info
        question = result_dict["conversation"]["question"]
        context = result_dict["conversation"]["context"]
        options = result_dict["conversation"].get("options", [])
        
        reply.text += f"ℹ️ **I need more information:**\n\n{question}\n\n"
        if options:
            reply.text += "**Options:**\n" + "\n".join(f"- {o}" for o in options) + "\n\n"
        reply.text += f"_{context}_"
        
        state.conversation_history.append(f"Assistant: {reply.text}")
        return reply, state
    
    if result_dict.get("choices"):
        # Tool recommendations
        choices = result_dict["choices"]
        state.last_choices = {c["name"]: c for c in choices}
        
        # Add all recommended tools to banlist for "another tool" queries
        for c in choices:
            if c.get("name"):
                state.banlist.add(c["name"])
        
        top_tool = choices[0]
        reply.text += f"✅ **I recommend {top_tool['name']}** ({top_tool.get('accuracy', 0):.1f}% match)\n\n"
        reply.text += f"_{top_tool.get('why', '')}_\n\n"
        
        # Format all choices as cards
        reply.text += "---\n\n"
        for i, choice in enumerate(choices, 1):
            tool_name = choice["name"]
            accuracy = choice.get("accuracy", 0.0)
            why = choice.get("why", "")
            
            # Get doc from index
            doc = doc_index.get(tool_name)
            if doc:
                card = format_tool_card(doc, accuracy, why, i)
                reply.text += card
                reply.text += "\n"
            else:
                reply.text += f"**{i}. {tool_name}** — {accuracy:.1f}%\n\n_{why}_\n\n"
        
        # Offer demo for top tool
        demo_url = top_tool.get("demo_link", "")
        if demo_url:
            state.pending_demo_tool = top_tool["name"]
            state.pending_demo_url = demo_url
            reply.text += f"\n💡 **Would you like me to run the demo for {top_tool['name']}?**\n"
            reply.text += f"🔗 Demo: {demo_url}\n\n"
            reply.text += "_Reply 'yes' to run the demo, or continue with another request._"
    else:
        # No suitable tools
        reason = result_dict.get("reason", "")
        explanation = result_dict.get("explanation", "")
        
        reply.text += f"❌ **No suitable tools found.**\n\n"
        if reason:
            reply.text += f"**Reason:** `{reason}`\n\n"
        if explanation:
            reply.text += explanation
    
    state.conversation_history.append(f"Assistant: {reply.text}")
    return reply, state
