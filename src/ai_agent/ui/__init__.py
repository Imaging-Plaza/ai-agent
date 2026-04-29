"""UI module for chat-based interface."""

from .state import ChatState, ChatMessage
from .formatters import format_tool_card


def create_chat_interface(*args, **kwargs):
    from .components import create_chat_interface as _create_chat_interface

    return _create_chat_interface(*args, **kwargs)


def launch(*args, **kwargs):
    from .app import launch as _launch

    return _launch(*args, **kwargs)


def get_pipeline(*args, **kwargs):
    from .app import get_pipeline as _get_pipeline

    return _get_pipeline(*args, **kwargs)


def refresh_ui_docs_from_index(*args, **kwargs):
    from .app import refresh_ui_docs_from_index as _refresh

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
