from __future__ import annotations

# ruff: noqa: E402 - tests add src/ to sys.path before importing the package.

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.agent.tools.mcp import get_tool, reload_registry, validate_config_payload
from ai_agent.agent.tools.mcp.registry import GRADIO_TOOLS_CONFIG_ENV, GenericGradioOutput
from ai_agent.services.chat import approve_pending, _shape_agent_result
from ai_agent.services.sessions import Session
from ai_agent.services.workflow_executor import execute_workflow
from ai_agent.services.workflow_planner import maybe_plan_workflow


def _endpoint(
    endpoint_id: str,
    *,
    aliases: list[str],
    input_role: str,
    output_role: str,
    output_name: str,
) -> dict:
    parameters = [
        {
            "name": "image",
            "source": "session_file",
            "file_index": 0,
            "required": True,
        }
    ]
    if output_role == "aligned":
        parameters.append(
            {
                "name": "mode",
                "source": "param",
                "param": "mode",
                "required": False,
                "value": "RIGID_BODY",
                "metadata": {"choices": ["TRANSLATION", "RIGID_BODY"]},
            }
        )
    return {
        "id": endpoint_id,
        "display_name": endpoint_id.replace("_", " ").title(),
        "api_name": f"/{endpoint_id}",
        "catalog_aliases": aliases,
        "input_mapping": {
            "parameters": parameters
        },
        "contracts": {
            "inputs": [
                {
                    "name": "image",
                    "artifact_type": "image.volume.3d",
                    "formats": ["tif", "tiff"],
                    "semantic_roles": [input_role],
                }
            ],
            "outputs": [
                {
                    "name": output_name,
                    "artifact_type": "image.volume.3d",
                    "formats": ["tif"],
                    "semantic_roles": [output_role],
                }
            ],
        },
    }


def _tool(tool_id: str, *, aliases: list[str], endpoint: dict) -> dict:
    return {
        "id": tool_id,
        "display_name": tool_id.replace("_", " ").title(),
        "enabled": True,
        "gradio_url": f"https://{tool_id}.hf.space/",
        "catalog_aliases": aliases,
        "default_endpoint": endpoint["id"],
        "endpoints": [endpoint],
    }


@pytest.fixture()
def workflow_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "stack_aligner",
                aliases=["stack alignment"],
                endpoint=_endpoint(
                    "align_stack",
                    aliases=["align stack"],
                    input_role="raw",
                    output_role="aligned",
                    output_name="aligned_stack",
                ),
            ),
            _tool(
                "lung_segmenter",
                aliases=["lung segmentation"],
                endpoint=_endpoint(
                    "segment_lungs",
                    aliases=["segment lungs"],
                    input_role="aligned",
                    output_role="mask",
                    output_name="lung_mask",
                ),
            ),
        ],
    }
    config_path = tmp_path / "gradio_tools.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(GRADIO_TOOLS_CONFIG_ENV, str(config_path))
    reload_registry(config_path)
    yield
    monkeypatch.delenv(GRADIO_TOOLS_CONFIG_ENV, raising=False)
    reload_registry()


def test_registry_accepts_optional_endpoint_contracts() -> None:
    config = validate_config_payload(
        {
            "version": 1,
            "tools": [
                _tool(
                    "contracted_tool",
                    aliases=["contracted"],
                    endpoint=_endpoint(
                        "run",
                        aliases=["contracted-run"],
                        input_role="raw",
                        output_role="mask",
                        output_name="mask",
                    ),
                )
            ],
        }
    )

    endpoint = config.tools[0].endpoints[0]
    assert endpoint.contracts.inputs[0].formats == ["tif", "tiff"]
    assert endpoint.contracts.outputs[0].semantic_roles == ["mask"]


def test_workflow_planner_finds_generic_compatible_chain(workflow_registry) -> None:
    plan = maybe_plan_workflow("align this stack then segment the lungs", ["scan.tif"])

    assert plan is not None
    assert [step.tool_name for step in plan.steps] == [
        "stack_aligner",
        "lung_segmenter",
    ]
    assert [step.output_name for step in plan.steps] == ["aligned_stack", "lung_mask"]


def test_workflow_planner_leaves_non_chain_request_alone(workflow_registry) -> None:
    assert maybe_plan_workflow("segment the lungs", ["scan.tif"]) is None


def test_chat_prefers_valid_workflow_over_format_clarification(workflow_registry) -> None:
    class FakeAgentResult:
        def to_legacy_dict(self):
            return {
                "tool_calls": [],
                "usage": None,
                "conversation": {
                    "status": "needs_clarification",
                    "question": "What file format is this stack?",
                    "options": ["DICOM", "NIfTI", "Other"],
                },
                "choices": [],
            }

    session = Session(session_id="session-1")
    result = _shape_agent_result(
        session,
        FakeAgentResult(),
        {},
        ["scan.tif"],
        "align this stack to the first frame and then segment the lungs",
    )

    assert result.status == "pending_action"
    assert result.pending_action is not None
    assert result.pending_action.type == "workflow_approval"
    assert session.pending_workflow_approval is not None
    assert result.pending_action.workflow_steps[0].runtime_parameters[0].name == "mode"


def test_workflow_executor_passes_artifact_between_steps(
    workflow_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "scan.tif"
    aligned = tmp_path / "aligned.tif"
    mask = tmp_path / "mask.tif"
    source.write_bytes(b"source")
    aligned.write_bytes(b"aligned")
    mask.write_bytes(b"mask")

    plan = maybe_plan_workflow("align this stack then segment the lungs", [str(source)])
    assert plan is not None
    seen_inputs: list[list[str]] = []
    seen_params: list[dict] = []

    def fake_ingest(session: Session, paths: list[str]):
        from ai_agent.services.files import FileIngestResult
        from ai_agent.services.sessions import Asset

        assets = []
        for path in paths:
            asset = Asset(asset_id=f"asset-{Path(path).stem}", path=path, display_name=Path(path).name)
            session.assets[asset.asset_id] = asset
            assets.append(asset)
        return FileIngestResult(assets=assets, validation_errors=[])

    def align_executor(inp):
        seen_inputs.append(list(inp.image_paths))
        seen_params.append(dict(inp.params))
        return GenericGradioOutput(success=True, result_origin=str(aligned))

    def segment_executor(inp):
        seen_inputs.append(list(inp.image_paths))
        seen_params.append(dict(inp.params))
        return GenericGradioOutput(success=True, result_origin=str(mask))

    monkeypatch.setattr("ai_agent.services.workflow_executor.ingest_files", fake_ingest)
    get_tool("stack_aligner", "align_stack").executor = align_executor
    get_tool("lung_segmenter", "segment_lungs").executor = segment_executor

    session = Session(session_id="session-1")
    plan.steps[0].params["mode"] = "TRANSLATION"
    result = execute_workflow(session, plan)

    assert result.success is True
    assert seen_inputs == [[str(source)], [str(aligned)]]
    assert seen_params == [{"mode": "TRANSLATION"}, {}]
    assert "input: scan.tif" in result.text
    assert "output: aligned.tif" in result.text
    assert "parameters: mode=TRANSLATION" in result.text
    assert len(result.files) == 2
    assert session.workflow_runs[-1]["success"] is True


def test_approve_pending_executes_workflow(
    workflow_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "scan.tif"
    aligned = tmp_path / "aligned.tif"
    mask = tmp_path / "mask.tif"
    for path in (source, aligned, mask):
        path.write_bytes(path.stem.encode())
    plan = maybe_plan_workflow("align this stack then segment the lungs", [str(source)])
    assert plan is not None

    def fake_ingest(session: Session, paths: list[str]):
        from ai_agent.services.files import FileIngestResult
        from ai_agent.services.sessions import Asset

        assets = [Asset(asset_id=f"asset-{Path(path).stem}", path=path) for path in paths]
        for asset in assets:
            session.assets[asset.asset_id] = asset
        return FileIngestResult(assets=assets, validation_errors=[])

    monkeypatch.setattr("ai_agent.services.workflow_executor.ingest_files", fake_ingest)
    seen_params: list[dict] = []

    def align_executor(inp):
        seen_params.append(dict(inp.params))
        return GenericGradioOutput(success=True, result_origin=str(aligned))

    get_tool("stack_aligner", "align_stack").executor = align_executor
    get_tool("lung_segmenter", "segment_lungs").executor = lambda inp: GenericGradioOutput(
        success=True, result_origin=str(mask)
    )

    session = Session(session_id="session-1")
    session.pending_workflow_approval = plan.id
    session.pending_workflow_plan = plan.to_dict()

    result = approve_pending(
        session,
        params={"workflow_steps": {"step_1": {"mode": "TRANSLATION"}}},
    )

    assert result.status == "tool_executed"
    assert "Workflow completed" in result.text
    assert seen_params == [{"mode": "TRANSLATION"}]
    assert session.pending_workflow_approval is None
