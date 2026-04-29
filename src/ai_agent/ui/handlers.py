import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

from ai_agent.agent.tools.gradio_space_tool import tool_run_example, RunExampleInput
from ai_agent.agent.tools.mcp import (
    get_tool,
    extract_preview,
    extract_downloads,
    extract_metadata,
    extract_output_field,
)
from ai_agent.retriever.software_doc import SoftwareDoc
from ai_agent.utils.file_validator import FileValidator
from ai_agent.utils.tags import strip_tags, parse_exclusions
from ai_agent.utils.previews import _build_preview_for_vlm
from ai_agent.utils.utils import _coerce_files_to_paths, _is_affirmative

from ai_agent.agent.agent import run_agent

from .state import ChatState, ChatMessage
from .formatters import format_tool_card

log = logging.getLogger("chat_handlers")


def execute_tool_with_approval(
    tool_name: str,
    tool_params: Dict[str, Any],
    state: ChatState,
) -> Tuple[ChatMessage, ChatState]:
    """
    Generic tool execution handler - works for ANY registered tool.

    Uses the tool registry to dynamically dispatch to the correct tool
    and extract results in a standardized way. No tool-specific code needed!

    Args:
        tool_name: Name of the tool to execute
        tool_params: Parameters for the tool
        state: Current chat state

    Returns:
        (ChatMessage with result, updated ChatState)
    """
    reply = ChatMessage()
    start_time = time.time()

    # Get tool configuration from registry
    tool_config = get_tool(tool_name)
    if not tool_config:
        log.error(f"Unknown tool: {tool_name}")
        reply.text = f"❌ Error: Unknown tool '{tool_name}'"
        state.pending_tool_approval = None
        state.pending_tool_params = {}
        return reply, state

    log.info(f"Executing {tool_name} tool with params: {tool_params}")
    reply.text = f"{tool_config.icon} Running {tool_config.display_name}...\n\n"

    try:
        # Augment params with state data if needed (e.g., image_path from last upload)
        if "image_path" in tool_params and not tool_params["image_path"]:
            if state.last_files:
                tool_params["image_path"] = state.last_files[0]

        # Build input object dynamically using the tool's input model
        input_obj = tool_config.input_model(**tool_params)

        # Execute the tool
        result = tool_config.executor(input_obj)

        compute_time = time.time() - start_time

        # Extract standard fields using registry configuration
        success = extract_output_field(result, tool_config.success_field)
        error = extract_output_field(result, tool_config.error_field)
        compute_time_seconds = (
            extract_output_field(result, tool_config.compute_time_field) or 0.0
        )
        notes = extract_output_field(result, tool_config.notes_field)

        # Track execution in state (generic)
        state.tool_calls.append(
            {
                "tool": tool_name,
                "success": success,
                "compute_time_seconds": compute_time_seconds,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                **tool_params,  # Store all params for debugging
            }
        )

        # Add stats to reply
        reply.stats = {
            "compute_time": compute_time_seconds,
            "total_time": compute_time,
        }

        if success:
            reply.text += f"✅ {tool_config.display_name} completed!\n\n"

            # Extract and add preview image (generic)
            preview_path = extract_preview(result, tool_name)
            if preview_path and os.path.exists(preview_path):
                reply.images.append(preview_path)

            # Extract and add downloadable files (generic)
            download_paths = extract_downloads(result, tool_name)
            for download_path in download_paths:
                if os.path.exists(download_path):
                    reply.files.append(
                        (download_path, f"Download {tool_config.display_name} result")
                    )

            # Add metadata if available
            metadata = extract_metadata(result, tool_name)
            if metadata:
                reply.text += f"_{metadata}_\n\n"

            # Add notes if available
            if notes:
                reply.text += f"_{notes}_\n\n"
        else:
            reply.text += f"❌ {tool_config.display_name} failed.\n\n"
            if error:
                reply.text += f"**Error:** {error}\n\n"

    except Exception as e:
        log.exception(f"Tool {tool_name} execution failed")
        reply.text += f"❌ Error: {e}\n\n"
        compute_time = time.time() - start_time
        reply.stats = {"total_time": compute_time}

    # Clear pending approval
    state.pending_tool_approval = None
    state.pending_tool_params = {}
    state.conversation_history.append(f"Assistant: {reply.text}")

    return reply, state


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
            demo_result = tool_run_example(
                RunExampleInput(
                    tool_name=state.pending_demo_tool,
                    image_path=pick,
                    endpoint_url=state.pending_demo_url or None,
                )
            )

            if demo_result.ran and (
                demo_result.result_preview or demo_result.result_image
            ):
                preview_path = demo_result.result_preview or demo_result.result_image
                reply.text += "✅ Demo completed!\n\n"
                reply.images.append(preview_path)

                # Add original result file for download if available
                if demo_result.result_origin:
                    reply.files.append((demo_result.result_origin, "Download result"))

            else:
                note = demo_result.notes or "No output image returned"
                reply.text += f"ℹ️ Demo ran but {note}"

            state.tool_calls.append(
                {
                    "tool": "run_example",
                    "tool_name": state.pending_demo_tool,
                    "ran": demo_result.ran,
                    "endpoint_url": demo_result.endpoint_url,
                    "timestamp": datetime.now().isoformat(),
                }
            )
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

    image_bytes = None
    if state.last_preview_path:
        try:
            # Read image bytes directly instead of converting to data URL
            preview_path = Path(state.last_preview_path)
            if preview_path.exists():
                image_bytes = preview_path.read_bytes()
                log.info(
                    f"✅ Image loaded: {len(image_bytes)} bytes from {state.last_preview_path}"
                )
                log.info("🖼️  Image will be sent to VLM as BinaryContent")
            else:
                log.warning(
                    f"⚠️ Preview path does not exist: {state.last_preview_path}"
                )
        except Exception as e:
            log.warning(
                "Failed to read image bytes from preview %r: %r",
                state.last_preview_path,
                e,
            )
    else:
        log.warning("⚠️ No preview path available - VLM will not receive image")

    # Extract original formats
    original_formats = []
    if file_paths:
        for pth in file_paths:
            ext = os.path.splitext(pth)[1].lower().lstrip(".")
            if ext == "gz" and pth.lower().endswith(".nii.gz"):
                ext = "nii.gz"
            if ext and ext not in original_formats:
                original_formats.append(ext)

    log.info(
        "Running agent: task=%s, formats=%s, excluded=%s",
        clean_message,
        original_formats,
        list(state.banlist),
    )

    # Parse model configuration if provided
    model_name = None
    base_url_override = None  # Use different variable name
    api_key_env = None
    if model:
        # Import here to avoid circular dependency
        from ai_agent.ui.components import get_model_config

        model_config = get_model_config(model)
        model_name = model_config.get("name")
        base_url_override = model_config.get("base_url")  # Can be None for OpenAI
        api_key_env = model_config.get("api_key_env", "OPENAI_API_KEY")
        log.info(
            f"Model config: {model} -> name={model_name}, base_url={base_url_override}, api_key_env={api_key_env}"
        )

    effective_paths = file_paths or (state.last_files or [])

    if not effective_paths:
        reply.text += (
            "⚠️ Please upload an image first (or re-upload). "
            "I need at least one image to recommend tools for your data."
        )
        state.conversation_history.append(f"Assistant: {reply.text}")
        return reply, state

    try:
        agent_result = run_agent(
            clean_message,
            image_paths=effective_paths,
            image_bytes=image_bytes,  # Pass image bytes to VLM
            excluded=list(state.banlist),
            conversation_history=state.conversation_history,
            model=model_name,
            base_url=(
                base_url_override if model else None
            ),  # Only override if model selected
            api_key_env=api_key_env,  # Pass the API key environment variable name
            top_k=top_k,
            num_choices=num_choices,
            image_metadata=state.last_image_meta,
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
            reply.text += (
                "💡 **Tip:** Set OPENAI_API_KEY in your .env file to use OpenAI models."
            )
        state.conversation_history.append(f"Assistant: {reply.text}")
        return reply, state
    except Exception as e:
        # Other errors (connection, API, etc.)
        error_msg = str(e)
        log.error(f"Agent execution error: {error_msg}", exc_info=True)
        reply.text = f"❌ **Error**\n\n{error_msg}\n\n"

        # Provide helpful hints based on error type
        if (
            "key_model_access_denied" in error_msg
            or "key not allowed" in error_msg.lower()
        ):
            reply.text += (
                "💡 **Tip:** This API key doesn't have access to this model.\n\n"
            )
            if "gpt-4o" in error_msg or "gpt-3.5" in error_msg:
                reply.text += "If using EPFL config, try selecting an EPFL model from the dropdown (e.g., 'openai/gpt-oss-120b [EPFL]')."
            else:
                reply.text += "If using OpenAI models, make sure you're using the correct API key."
        elif "ConnectError" in error_msg or "Connection" in error_msg:
            reply.text += "💡 **Tip:** Connection failed. If using EPFL models, ensure you're connected to EPFL VPN."

        state.conversation_history.append(f"Assistant: {reply.text}")
        return reply, state

    result_dict = agent_result.to_legacy_dict()

    # Extract usage stats if available
    usage_info = result_dict.get("usage")
    if usage_info:
        reply.stats = {
            "tokens": {
                "total": usage_info.get("total_tokens", 0),
                "input": usage_info.get("input_tokens", 0),
                "output": usage_info.get("output_tokens", 0),
            }
        }

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
            reply.text += (
                "**Options:**\n" + "\n".join(f"- {o}" for o in options) + "\n\n"
            )
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

        # Check if top tool is registered in registry and requires approval
        tool_config = get_tool(top_tool["name"])
        demo_url = top_tool.get("demo_link") or ""

        if tool_config and tool_config.requires_approval:
            # Tool is registered and requires approval - use registry-based execution
            image_path = effective_paths[0] if effective_paths else None
            state.pending_tool_approval = tool_config.name
            state.pending_tool_params = {
                "image_path": image_path,
                "description": f"Recommended by agent: {top_tool.get('why', '')}",
            }
            reply.text += f"\n🚀 **Ready to run {tool_config.display_name}?**\n\n"
            reply.text += f"📁 **Image:** {os.path.basename(image_path) if image_path else 'Unknown'}\n"
            if demo_url:
                reply.text += f"🔗 **Endpoint:** {demo_url}\n\n"
            reply.text += f"_Press the **'{tool_config.icon} Run Tool'** button below, or ask about other tools in the chat instead._"
        elif demo_url:
            # Tool has demo but not registered - use generic demo flow
            state.pending_demo_tool = top_tool["name"]
            state.pending_demo_url = demo_url
            reply.text += (
                f"\n💡 **Would you like me to run the demo for {top_tool['name']}?**\n"
            )
            reply.text += f"🔗 Demo: {demo_url}\n\n"
            reply.text += "_Press the **'🚀 Run Demo'** button to run the demo, or continue with another request._"
    else:
        # No suitable tools
        reason = result_dict.get("reason", "")
        explanation = result_dict.get("explanation", "")

        reply.text += "❌ **No suitable tools found.**\n\n"
        if reason:
            reply.text += f"**Reason:** `{reason}`\n\n"
        if explanation:
            reply.text += explanation

    state.conversation_history.append(f"Assistant: {reply.text}")
    return reply, state
