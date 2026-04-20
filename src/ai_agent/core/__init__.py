"""Core shared services and singletons.

This package contains all frontend-agnostic business logic:

* ``chat_state``    – ChatState / ChatMessage data models
* ``handlers``      – respond() / execute_tool_with_approval()
* ``formatters``    – format_tool_card()
* ``model_config``  – model configuration lookup
* ``app_setup``     – logging, pipeline init, doc index refresh
* ``pipeline_registry`` – RAGImagingPipeline singleton

Imports are kept lazy to avoid heavy transitive loads at package level.
Use explicit imports from the submodules, e.g.::

    from ai_agent.core.chat_state import ChatState
    from ai_agent.core.handlers import respond
"""

from .pipeline_registry import get_pipeline, reset_pipeline

__all__ = [
    "get_pipeline",
    "reset_pipeline",
]
