"""Transport-agnostic application services.

These modules contain the orchestration logic that used to live inline in
``ai_agent.ui.handlers``. They are designed to be callable from either the
legacy Gradio UI or the new FastAPI backend without any UI dependencies.
"""
