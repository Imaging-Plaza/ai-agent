from __future__ import annotations

# ruff: noqa: E402 - tests add src/ to sys.path before importing the package.

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src"
for p in (ROOT, PKG_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from ai_agent.agent.models import AgentToolSelection, ToolRunLog
from ai_agent.agent.tools.mcp import reload_registry, validate_config_payload
from ai_agent.agent.tools.mcp.registry import (
    GRADIO_TOOLS_CONFIG_ENV,
    GenericGradioOutput,
    RegistryValidationError,
    TOOL_REGISTRY,
)
from ai_agent.generator.schema import Conversation, ConversationStatus, ToolChoice
from ai_agent.services import chat as chat_service
from ai_agent.services.chat import ChatRequest, approve_pending, decline_pending, process_turn
from ai_agent.services.sessions import Asset, Session


def _agent_result(
    *choices: str,
    tool_calls: list[ToolRunLog] | None = None,
) -> AgentToolSelection:
    return AgentToolSelection(
        conversation=Conversation(status=ConversationStatus.COMPLETE),
        choices=[
            ToolChoice(
                name=name,
                rank=i,
                accuracy=90.0 - i,
                why=f"{name} matches the request",
            )
            for i, name in enumerate(choices, 1)
        ],
        tool_calls=tool_calls or [],
    )


def _no_tool_result(explanation: str = "No endpoint could process this.") -> AgentToolSelection:
    return AgentToolSelection(
        conversation=Conversation(status=ConversationStatus.COMPLETE),
        choices=[],
        explanation=explanation,
        reason="no_suitable_tool",
    )


@pytest.fixture()
def configured_mcp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = {
        "version": 1,
        "tools": [
            {
                "id": "demo_tool",
                "display_name": "Demo Tool",
                "enabled": True,
                "gradio_url": "https://example.com/",
                "catalog_aliases": ["demo-tool"],
                "default_endpoint": "run",
                "endpoints": [
                    {
                        "id": "run",
                        "display_name": "Run Demo Tool",
                        "api_name": "/run",
                        "catalog_aliases": ["demo-tool-run"],
                        "input_mapping": {
                            "parameters": [
                                {
                                    "name": "image",
                                    "source": "session_file",
                                    "required": True,
                                }
                            ]
                        },
                    }
                ],
            }
        ],
    }
    config_path = tmp_path / "gradio_tools.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(GRADIO_TOOLS_CONFIG_ENV, str(config_path))
    reload_registry(config_path)
    yield
    monkeypatch.delenv(GRADIO_TOOLS_CONFIG_ENV, raising=False)
    reload_registry()


def _asset_session(tmp_path: Path, *, with_preview: bool = True) -> Session:
    image_path = tmp_path / "scan.tif"
    image_path.write_bytes(b"fake-tiff")
    preview_path = tmp_path / "preview.png"
    if with_preview:
        preview_path.write_bytes(b"fake-png")
    session = Session(session_id="session-1")
    session.assets["asset-1"] = Asset(
        asset_id="asset-1",
        path=str(image_path),
        preview_path=str(preview_path) if with_preview else None,
        metadata_text="TIFF stack, 2 frames",
        original_format="tif",
    )
    return session


def test_normal_text_only_chat_calls_agent_without_image_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_agent(task: str, image_paths: list[str], **kwargs):
        captured.update({"task": task, "image_paths": image_paths, **kwargs})
        return _agent_result("text-tool")

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)
    session = Session(session_id="text-only")

    result = process_turn(
        session,
        ChatRequest(message="Find denoising tools"),
        doc_index={},
    )

    assert result.status == "ok"
    assert result.recommendations[0].name == "text-tool"
    assert captured["image_paths"] == []
    assert captured["image_bytes"] is None
    assert captured["image_metadata"] is None
    assert session.conversation_history[0] == "User: Find denoising tools"
    assert session.conversation_history[-1].startswith("Assistant:")


def test_chat_with_uploaded_image_passes_paths_preview_bytes_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _asset_session(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run_agent(task: str, image_paths: list[str], **kwargs):
        captured.update({"task": task, "image_paths": image_paths, **kwargs})
        return _agent_result("image-tool")

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)

    result = process_turn(
        session,
        ChatRequest(message="Segment this image", asset_ids=["asset-1"]),
        doc_index={},
    )

    assert result.status == "ok"
    assert captured["image_paths"] == [session.assets["asset-1"].path]
    assert captured["image_bytes"] == b"fake-png"
    assert captured["image_metadata"] == "TIFF stack, 2 frames"
    assert session.last_asset_ids == ["asset-1"]


def test_tool_search_selection_creates_pending_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_mcp_registry
) -> None:
    session = _asset_session(tmp_path)

    def fake_run_agent(*args, **kwargs):
        return _agent_result(
            "demo-tool",
            tool_calls=[
                ToolRunLog(
                    tool="search_tools",
                    inputs={"query": "segment lungs", "count": 1},
                )
            ],
        )

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)

    result = process_turn(
        session,
        ChatRequest(message="Segment this image", asset_ids=["asset-1"]),
        doc_index={},
    )

    assert result.status == "pending_action"
    assert result.pending_action is not None
    assert result.pending_action.tool_name == "demo_tool"
    assert result.pending_action.endpoint_id == "run"
    assert session.pending_tool_approval == "demo_tool"
    assert session.pending_tool_params["image_path"] == session.assets["asset-1"].path
    assert session.tool_calls[0]["tool"] == "search_tools"


def test_catalog_demo_link_without_upload_does_not_create_tool_approval(
    monkeypatch: pytest.MonkeyPatch, configured_mcp_registry
) -> None:
    def fake_run_agent(*args, **kwargs):
        return AgentToolSelection(
            conversation=Conversation(status=ConversationStatus.COMPLETE),
            choices=[
                ToolChoice(
                    name="demo-tool",
                    rank=1,
                    accuracy=99.0,
                    why="Catalog has a runnable demo, but no session file is active.",
                    demo_link="https://example.com/catalog-demo",
                )
            ],
        )

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)
    session = Session(session_id="no-upload")

    result = process_turn(
        session,
        ChatRequest(message="Can I run demo-tool?"),
        doc_index={},
    )

    assert result.status == "ok"
    assert result.pending_action is None
    assert result.recommendations[0].demo_url == "https://example.com/catalog-demo"
    assert session.pending_tool_approval is None


def test_last_uploaded_asset_enables_tool_approval_when_turn_has_no_asset_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_mcp_registry
) -> None:
    session = _asset_session(tmp_path)
    session.last_asset_ids = ["asset-1"]

    def fake_run_agent(task: str, image_paths: list[str], **kwargs):
        assert image_paths == [session.assets["asset-1"].path]
        return _agent_result("demo-tool")

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)

    result = process_turn(
        session,
        ChatRequest(message="Run this on the same image"),
        doc_index={},
    )

    assert result.status == "pending_action"
    assert result.pending_action is not None
    assert result.pending_action.image_name == "scan.tif"
    assert session.pending_tool_params["image_path"] == session.assets["asset-1"].path


def test_endpoint_specific_alias_creates_pending_action_for_matching_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_mcp_registry
) -> None:
    session = _asset_session(tmp_path)

    def fake_run_agent(*args, **kwargs):
        return _agent_result("demo-tool-run")

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)

    result = process_turn(
        session,
        ChatRequest(message="Use the explicit runnable endpoint", asset_ids=["asset-1"]),
        doc_index={},
    )

    assert result.status == "pending_action"
    assert result.pending_action is not None
    assert result.pending_action.endpoint_id == "run"
    assert result.pending_action.matched_alias == "demo-tool-run"
    assert session.pending_catalog_alias == "demo-tool-run"


def test_approval_executes_tool_and_records_success_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_mcp_registry
) -> None:
    session = _asset_session(tmp_path)
    session.pending_tool_approval = "demo_tool"
    session.pending_tool_endpoint = "run"
    session.pending_recommendation_name = "demo-tool"
    session.pending_recommendation_rank = 1
    session.pending_catalog_alias = "demo-tool"
    session.pending_tool_params = {
        "endpoint_id": "run",
        "image_path": session.assets["asset-1"].path,
        "image_paths": [session.assets["asset-1"].path],
        "description": "Recommended by agent",
    }
    preview_path = tmp_path / "result.png"
    preview_path.write_bytes(b"png")

    def fake_executor(inp):
        return GenericGradioOutput(
            success=True,
            result_preview=str(preview_path),
            result_origin=str(preview_path),
            metadata_text="mask voxels=10",
            notes="finished",
            compute_time_seconds=0.25,
        )

    monkeypatch.setattr(chat_service, "ingest_files", lambda session, paths: _fake_ingest(session, paths))
    TOOL_REGISTRY["demo_tool:run"].executor = fake_executor

    result = approve_pending(session)

    assert result.status == "tool_executed"
    assert "completed" in result.text.lower()
    assert "mask voxels=10" in result.text
    assert session.pending_tool_approval is None
    assert session.tool_calls[-1]["success"] is True
    assert session.conversation_history[-1].startswith("Assistant:")


def test_rejection_clears_pending_action_and_records_history(
    tmp_path: Path, configured_mcp_registry
) -> None:
    session = _asset_session(tmp_path)
    session.pending_tool_approval = "demo_tool"
    session.pending_tool_endpoint = "run"
    session.pending_tool_params = {"endpoint_id": "run"}

    result = decline_pending(session)

    assert result.status == "ok"
    assert session.pending_tool_approval is None
    assert session.pending_tool_endpoint is None
    assert session.pending_tool_params == {}
    assert session.conversation_history[-1].startswith("Assistant:")


def test_endpoint_failure_records_failed_tool_call_and_history(
    tmp_path: Path, configured_mcp_registry
) -> None:
    session = _asset_session(tmp_path)
    session.pending_tool_approval = "demo_tool"
    session.pending_tool_endpoint = "run"
    session.pending_tool_params = {
        "endpoint_id": "run",
        "image_path": session.assets["asset-1"].path,
        "image_paths": [session.assets["asset-1"].path],
    }

    def fake_executor(inp):
        return GenericGradioOutput(
            success=False,
            error="endpoint exploded",
            compute_time_seconds=0.1,
        )

    TOOL_REGISTRY["demo_tool:run"].executor = fake_executor

    result = approve_pending(session)

    assert result.status == "tool_executed"
    assert "failed" in result.text.lower()
    assert "endpoint exploded" in result.text
    assert session.pending_tool_approval is None
    assert session.tool_calls[-1]["success"] is False
    assert session.tool_calls[-1]["error"] == "endpoint exploded"
    assert session.conversation_history[-1].startswith("Assistant:")


def test_invalid_mcp_configuration_reports_validation_error() -> None:
    payload = {
        "version": 1,
        "tools": [
            {
                "id": "secret_tool",
                "display_name": "Secret Tool",
                "enabled": True,
                "gradio_url": "https://example.com/",
                "auth": {"token_env": "sk-this-is-a-token-not-an-env-name"},
                "endpoints": [
                    {
                        "id": "run",
                        "display_name": "Run",
                        "api_name": "/run",
                        "input_mapping": {"parameters": [{"name": "image"}]},
                    }
                ],
            }
        ],
    }

    with pytest.raises(RegistryValidationError, match="environment variable name"):
        validate_config_payload(payload)


def test_failed_agent_turn_records_user_and_error_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_agent(*args, **kwargs):
        return _no_tool_result("Nothing matched this request.")

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)
    session = Session(session_id="failure")

    result = process_turn(
        session,
        ChatRequest(message="Find a tool for impossible task"),
        doc_index={},
    )

    assert result.status == "no_results"
    assert session.conversation_history[0] == "User: Find a tool for impossible task"
    assert session.conversation_history[-1].startswith("Assistant:")
    assert "No suitable tools found" in session.conversation_history[-1]


def test_current_message_is_not_sent_twice_to_agent_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_agent(task: str, image_paths: list[str], **kwargs):
        captured.update(
            {
                "task": task,
                **{
                    key: list(value) if key == "conversation_history" else value
                    for key, value in kwargs.items()
                },
            }
        )
        return _agent_result("text-tool")

    monkeypatch.setattr(chat_service, "run_agent", fake_run_agent)
    session = Session(session_id="duplicate-check")

    process_turn(
        session,
        ChatRequest(message="Segment these nuclei"),
        doc_index={},
    )

    history_text = "\n".join(captured["conversation_history"])
    assert captured["task"] == "Segment these nuclei"
    assert "Segment these nuclei" not in history_text


def test_text_only_agent_prompt_does_not_include_image_specific_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_agent.agent import agent as agent_module

    captured: dict[str, Any] = {}

    class FakeUsage:
        total_tokens = 1
        input_tokens = 1
        output_tokens = 0

    class FakeRunResult:
        output = _agent_result("text-tool")

        def usage(self):
            return FakeUsage()

    class FakeAgent:
        def run_sync(self, user_prompt, **kwargs):
            captured["user_prompt"] = user_prompt
            return FakeRunResult()

    monkeypatch.setattr(agent_module, "agent", FakeAgent())
    monkeypatch.setattr(agent_module, "summarize_image_metadata", lambda paths: "")
    monkeypatch.setattr(agent_module, "detect_ext_token", lambda paths: "")

    agent_module.run_agent(
        "Find denoising tools",
        image_paths=[],
        conversation_history=[],
    )

    assert isinstance(captured["user_prompt"], str)
    assert "attached preview image" not in captured["user_prompt"].casefold()
    assert "visual observations" not in captured["user_prompt"].casefold()


def _fake_ingest(session: Session, paths: list[str]):
    from ai_agent.services.files import FileIngestResult

    assets = []
    for i, path in enumerate(paths):
        asset = Asset(
            asset_id=f"artifact-{i}",
            path=path,
            preview_path=path,
            display_name=Path(path).name,
        )
        session.assets[asset.asset_id] = asset
        assets.append(asset)
    return FileIngestResult(assets=assets, validation_errors=[])
