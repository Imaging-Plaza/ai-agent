"""Core shared services and singletons."""

from .pipeline_registry import get_pipeline, reset_pipeline

__all__ = ["get_pipeline", "reset_pipeline"]
