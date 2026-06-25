from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai_agent.agent.tools.mcp import extract_downloads, extract_metadata, extract_output_field, extract_preview, get_tool
from .files import ingest_files
from .sessions import Asset, Session
from .workflow_planner import PlannedWorkflow

log = logging.getLogger("services.workflow_executor")


@dataclass
class WorkflowExecutionResult:
    success: bool
    text: str
    images: List[str] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    traces: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


def execute_workflow(session: Session, plan: PlannedWorkflow) -> WorkflowExecutionResult:
    started = time.time()
    current_paths = list(plan.input_paths)
    images: List[str] = []
    files: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    text_parts = [f"Running {plan.display_name}...\n"]

    for index, step in enumerate(plan.steps, 1):
        tool_config = get_tool(step.tool_name, step.endpoint_id)
        if not tool_config or not tool_config.endpoint:
            error = f"Unknown workflow step endpoint: {step.tool_name}/{step.endpoint_id}"
            return _workflow_failure(session, plan, text_parts, traces, error)
        if not tool_config.is_runnable():
            error = f"Workflow step is not runnable: {tool_config.display_name}"
            return _workflow_failure(session, plan, text_parts, traces, error)
        if not current_paths:
            error = f"Workflow step {index} has no input artifact"
            return _workflow_failure(session, plan, text_parts, traces, error)

        step_started = time.time()
        input_obj = tool_config.input_model(
            tool_id=tool_config.name,
            endpoint_id=tool_config.endpoint.id,
            image_path=current_paths[0],
            image_paths=list(current_paths),
            description=f"Workflow step {index}/{len(plan.steps)}: {step.operation}",
            params=dict(step.params),
        )
        try:
            result = tool_config.executor(input_obj)
        except Exception as exc:
            log.exception("Workflow step %s failed", step.id)
            error = f"{tool_config.display_name} failed: {exc}"
            return _workflow_failure(session, plan, text_parts, traces, error)

        success = bool(extract_output_field(result, tool_config.success_field))
        error = extract_output_field(result, tool_config.error_field)
        compute_time = extract_output_field(result, tool_config.compute_time_field) or (
            time.time() - step_started
        )
        preview_path = extract_preview(result, tool_config.name)
        downloads = extract_downloads(result, tool_config.name)
        metadata = extract_metadata(result, tool_config.name)

        trace = {
            "tool": tool_config.name,
            "endpoint": tool_config.endpoint.id,
            "workflow": plan.id,
            "workflow_step": step.id,
            "workflow_step_index": index,
            "operation": step.operation,
            "success": success,
            "compute_time_seconds": compute_time,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "image_paths": list(current_paths),
            "params": dict(step.params),
        }
        traces.append(trace)
        session.tool_calls.append(trace)

        if not success:
            message = str(error or f"{tool_config.display_name} returned an unsuccessful response")
            return _workflow_failure(session, plan, text_parts, traces, message)

        artifact_paths = [path for path in downloads if path and os.path.exists(path)]
        if preview_path and os.path.exists(preview_path):
            asset = _register_workflow_artifact(session, preview_path)
            if asset and asset.preview_path:
                images.append(_asset_preview_url(asset))
        registered_files: List[Dict[str, Any]] = []
        for artifact_path in artifact_paths:
            asset = _register_workflow_artifact(session, artifact_path)
            if not asset:
                continue
            item = {
                "path": _asset_raw_url(asset),
                "label": f"{step.endpoint_display_name} result",
                "asset_id": asset.asset_id,
                "preview_url": _asset_preview_url(asset) if asset.preview_path else None,
                "display_name": asset.display_name,
                "workflow_step": step.id,
                "output_name": step.output_name,
            }
            registered_files.append(item)
            files.append(item)

        primary_output = artifact_paths[0] if artifact_paths else preview_path
        if not primary_output:
            message = f"{tool_config.display_name} completed but did not produce a chainable artifact"
            return _workflow_failure(session, plan, text_parts, traces, message)

        current_paths = [primary_output]
        text_parts.append(f"{index}. {step.endpoint_display_name} completed.")
        text_parts.append(f"   input: {_display_path(trace['image_paths'][0])}")
        text_parts.append(f"   output: {_display_path(primary_output)}")
        if step.params:
            text_parts.append(f"   parameters: {_format_params(step.params)}")
        if metadata:
            text_parts.append(f"   {metadata}")

    elapsed = time.time() - started
    session.workflow_runs.append(
        {
            "id": plan.id,
            "display_name": plan.display_name,
            "success": True,
            "step_count": len(plan.steps),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
    )
    text_parts.append(f"\nWorkflow completed in {elapsed:.2f}s.")
    return WorkflowExecutionResult(
        success=True,
        text="\n".join(text_parts),
        images=images,
        files=files,
        traces=traces,
    )


def _workflow_failure(
    session: Session,
    plan: PlannedWorkflow,
    text_parts: List[str],
    traces: List[Dict[str, Any]],
    error: str,
) -> WorkflowExecutionResult:
    session.workflow_runs.append(
        {
            "id": plan.id,
            "display_name": plan.display_name,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
    )
    text_parts.append(f"Workflow stopped: {error}")
    return WorkflowExecutionResult(
        success=False,
        text="\n".join(text_parts),
        traces=traces,
        error=error,
    )


def _register_workflow_artifact(session: Session, path: str) -> Optional[Asset]:
    previous_last_asset_ids = list(session.last_asset_ids)
    try:
        result = ingest_files(session, [path])
    except Exception:
        log.exception("Workflow artifact registration failed for %s", path)
        return None
    finally:
        session.last_asset_ids = previous_last_asset_ids
        session.touch()
    if result.validation_errors:
        log.warning("Workflow artifact validation failed for %s: %s", path, result.validation_errors)
    return result.assets[0] if result.assets else None


def _asset_preview_url(asset: Asset) -> str:
    return f"/api/files/preview/{asset.asset_id}"


def _asset_raw_url(asset: Asset) -> str:
    return f"/api/files/asset/{asset.asset_id}/raw"


def _display_path(path: str) -> str:
    return os.path.basename(path) or path


def _format_params(params: Dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


__all__ = ["WorkflowExecutionResult", "execute_workflow"]
