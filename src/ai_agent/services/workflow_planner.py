from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai_agent.agent.tools.mcp.registry import ArtifactContract, ToolConfig, list_tool_endpoints, list_tools


@dataclass
class PlannedWorkflowStep:
    id: str
    tool_name: str
    endpoint_id: str
    display_name: str
    endpoint_display_name: str
    input_name: str
    output_name: str
    operation: str
    description: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannedWorkflow:
    id: str
    display_name: str
    prompt: str
    steps: List[PlannedWorkflowStep] = field(default_factory=list)
    input_paths: List[str] = field(default_factory=list)
    request_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "prompt": self.prompt,
            "input_paths": list(self.input_paths),
            "request_text": self.request_text,
            "steps": [step.__dict__.copy() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannedWorkflow":
        return cls(
            id=str(payload.get("id") or "workflow"),
            display_name=str(payload.get("display_name") or "Tool chain"),
            prompt=str(payload.get("prompt") or ""),
            input_paths=list(payload.get("input_paths") or []),
            request_text=str(payload.get("request_text") or ""),
            steps=[
                PlannedWorkflowStep(
                    id=str(item.get("id") or f"step_{i + 1}"),
                    tool_name=str(item.get("tool_name") or ""),
                    endpoint_id=str(item.get("endpoint_id") or ""),
                    display_name=str(item.get("display_name") or ""),
                    endpoint_display_name=str(item.get("endpoint_display_name") or ""),
                    input_name=str(item.get("input_name") or "input"),
                    output_name=str(item.get("output_name") or "output"),
                    operation=str(item.get("operation") or ""),
                    description=item.get("description"),
                    params=dict(item.get("params") or {}),
                )
                for i, item in enumerate(payload.get("steps") or [])
                if isinstance(item, dict)
            ],
        )


@dataclass
class _Candidate:
    tool: ToolConfig
    input_contract: ArtifactContract
    output_contract: ArtifactContract
    operations: set[str]


_CHAIN_MARKERS = (
    " then ",
    " after ",
    " before ",
    " followed by ",
    " and then ",
    " pipeline",
    " workflow",
    "chain",
)

_OPERATION_ALIASES: Dict[str, tuple[str, ...]] = {
    "align": ("align", "aligned", "alignment", "register", "registration", "stabilize", "drift"),
    "convert": ("convert", "conversion", "export", "reformat", "transform format"),
    "denoise": ("denoise", "denoising", "noise", "restore", "restoration", "deblur"),
    "segment": ("segment", "segmentation", "mask", "label", "partition"),
    "quantify": ("quantify", "measure", "measurement", "analyze", "analyse", "count"),
    "track": ("track", "tracking"),
    "visualize": ("visualize", "visualise", "view", "display", "render"),
}


def maybe_plan_workflow(request_text: str, input_paths: List[str]) -> Optional[PlannedWorkflow]:
    """Return a generic linear chain when endpoint contracts prove compatibility."""
    operations = _requested_operations(request_text)
    if len(operations) < 2:
        return None
    if not _looks_like_chain_request(request_text, operations):
        return None

    candidates = _contracted_candidates()
    if not candidates:
        return None

    current = _initial_contract(input_paths)
    steps: List[PlannedWorkflowStep] = []
    used: set[tuple[str, str]] = set()
    for operation in operations[:3]:
        match = _best_candidate(operation, current, candidates, used)
        if match is None:
            return None
        tool = match.tool
        endpoint = tool.endpoint
        if endpoint is None:
            return None
        used.add((tool.name, endpoint.id))
        steps.append(
            PlannedWorkflowStep(
                id=f"step_{len(steps) + 1}",
                tool_name=tool.name,
                endpoint_id=endpoint.id,
                display_name=tool.display_name,
                endpoint_display_name=endpoint.display_name,
                input_name=match.input_contract.name,
                output_name=match.output_contract.name,
                operation=operation,
                description=endpoint.description,
            )
        )
        current = match.output_contract

    if len(steps) < 2:
        return None
    label = " -> ".join(step.endpoint_display_name for step in steps)
    return PlannedWorkflow(
        id=_workflow_id(steps),
        display_name=label,
        prompt=f"Run this {len(steps)}-step tool chain?",
        steps=steps,
        input_paths=list(input_paths),
        request_text=request_text,
    )


def contracts_compatible(source: ArtifactContract, target: ArtifactContract) -> bool:
    if not _artifact_type_compatible(source.artifact_type, target.artifact_type):
        return False
    source_formats = set(source.formats)
    target_formats = set(target.formats)
    if source_formats and target_formats and source_formats.isdisjoint(target_formats):
        return False
    source_roles = set(source.semantic_roles)
    target_roles = set(target.semantic_roles)
    if target_roles and source_roles and source_roles.isdisjoint(target_roles):
        # Roles are hints: incompatible roles should reduce chaining only when
        # both sides explicitly constrain them.
        return False
    return True


def _contracted_candidates() -> List[_Candidate]:
    candidates: List[_Candidate] = []
    for tool_name in list_tools():
        for tool in list_tool_endpoints(tool_name):
            endpoint = tool.endpoint
            if not endpoint or not tool.is_runnable():
                continue
            for inp in endpoint.contracts.inputs:
                for out in endpoint.contracts.outputs:
                    candidates.append(
                        _Candidate(
                            tool=tool,
                            input_contract=inp,
                            output_contract=out,
                            operations=_candidate_operations(tool, inp, out),
                        )
                    )
    return candidates


def _best_candidate(
    operation: str,
    current: ArtifactContract,
    candidates: List[_Candidate],
    used: set[tuple[str, str]],
) -> Optional[_Candidate]:
    scored: List[tuple[float, _Candidate]] = []
    for candidate in candidates:
        endpoint = candidate.tool.endpoint
        if endpoint is None or (candidate.tool.name, endpoint.id) in used:
            continue
        if operation not in candidate.operations:
            continue
        if not contracts_compatible(current, candidate.input_contract):
            continue
        scored.append((_candidate_score(operation, current, candidate), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _candidate_score(operation: str, current: ArtifactContract, candidate: _Candidate) -> float:
    score = 1.0
    if operation in candidate.operations:
        score += 5.0
    if set(current.formats) & set(candidate.input_contract.formats):
        score += 1.5
    if set(current.semantic_roles) & set(candidate.input_contract.semantic_roles):
        score += 1.0
    if candidate.output_contract.semantic_roles:
        score += 0.25
    return score


def _requested_operations(request_text: str) -> List[str]:
    text = _normalize(request_text)
    positions: List[tuple[int, str]] = []
    for operation, aliases in _OPERATION_ALIASES.items():
        indexes = [text.find(alias) for alias in aliases if alias in text]
        indexes = [idx for idx in indexes if idx >= 0]
        if indexes:
            positions.append((min(indexes), operation))
    positions.sort(key=lambda item: item[0])
    ordered: List[str] = []
    for _, operation in positions:
        if operation not in ordered:
            ordered.append(operation)
    return ordered


def _looks_like_chain_request(request_text: str, operations: List[str]) -> bool:
    text = f" {_normalize(request_text)} "
    return len(operations) >= 2 and any(marker in text for marker in _CHAIN_MARKERS)


def _candidate_operations(tool: ToolConfig, inp: ArtifactContract, out: ArtifactContract) -> set[str]:
    endpoint = tool.endpoint
    parts = [
        tool.name,
        tool.display_name,
        endpoint.id if endpoint else "",
        endpoint.display_name if endpoint else "",
        endpoint.description if endpoint else "",
        " ".join(tool.catalog_names or []),
        " ".join(out.semantic_roles),
        out.name,
    ]
    text = _normalize(" ".join(p for p in parts if p))
    operations = {
        operation
        for operation, aliases in _OPERATION_ALIASES.items()
        if any(alias in text for alias in aliases)
    }
    return operations


def _initial_contract(input_paths: List[str]) -> ArtifactContract:
    formats = [_format_from_path(path) for path in input_paths]
    return ArtifactContract(
        name="uploaded_input",
        artifact_type="file",
        formats=[fmt for fmt in formats if fmt],
        semantic_roles=["raw", "input"],
    )


def _artifact_type_compatible(source: str, target: str) -> bool:
    source_norm = _normalize_type(source)
    target_norm = _normalize_type(target)
    if source_norm in {"file", "any"} or target_norm in {"file", "any"}:
        return True
    return source_norm == target_norm or source_norm.startswith(target_norm) or target_norm.startswith(source_norm)


def _format_from_path(path: str) -> str:
    lower = (path or "").casefold()
    if lower.endswith(".nii.gz"):
        return "nii.gz"
    return os.path.splitext(lower)[1].lstrip(".")


def _workflow_id(steps: List[PlannedWorkflowStep]) -> str:
    raw = "_then_".join(f"{step.tool_name}_{step.endpoint_id}" for step in steps)
    return re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_") or "tool_chain"


def _normalize(value: str) -> str:
    return " ".join((value or "").replace("_", " ").replace("-", " ").casefold().split())


def _normalize_type(value: str) -> str:
    return (value or "file").strip().replace("_", ".").replace("-", ".").casefold()


__all__ = [
    "PlannedWorkflow",
    "PlannedWorkflowStep",
    "contracts_compatible",
    "maybe_plan_workflow",
]
