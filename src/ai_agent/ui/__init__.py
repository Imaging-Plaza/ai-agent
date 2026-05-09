"""Gradio chat interface."""


def get_pipeline(*args, **kwargs):
    from ai_agent.core.app_setup import get_pipeline as _get_pipeline

    return _get_pipeline(*args, **kwargs)


def refresh_ui_docs_from_index(*args, **kwargs):
    from ai_agent.core.app_setup import refresh_docs_from_index as _refresh

    return _refresh(*args, **kwargs)

__all__ = [
    "ChatState",
    "ChatMessage",
    "format_tool_card",
    "create_chat_interface",
    "launch",
    "get_pipeline",
    "refresh_ui_docs_from_index",
]
