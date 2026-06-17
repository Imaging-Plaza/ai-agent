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

from ai_agent.agent.tools.mcp import (
    RegistryValidationError,
    alias_is_tool_level,
    list_tool_endpoints,
    list_tools,
    reload_registry,
    resolve_catalog_alias,
    validate_config_payload,
)
from ai_agent.agent.tools.mcp.registry import GRADIO_TOOLS_CONFIG_ENV
from ai_agent.services.chat import _select_pending_action
from ai_agent.services.sessions import Asset, Session


def _endpoint(
    endpoint_id: str,
    *,
    aliases: list[str],
    api_name: str | None = None,
    demo_available: bool = True,
    file_index: int = 0,
) -> dict:
    return {
        "id": endpoint_id,
        "display_name": endpoint_id.replace("_", " ").title(),
        "api_name": api_name or f"/{endpoint_id}",
        "catalog_aliases": aliases,
        "input_mapping": {
            "parameters": [
                {
                    "name": "image",
                    "source": "session_file",
                    "file_index": file_index,
                    "required": True,
                }
            ]
        },
        "demo": {"available": demo_available},
    }


def _tool(
    tool_id: str,
    *,
    aliases: list[str],
    endpoints: list[dict],
    default_endpoint: str | None = None,
    url: str = "https://example.com/",
) -> dict:
    return {
        "id": tool_id,
        "display_name": tool_id.replace("_", " ").title(),
        "enabled": True,
        "gradio_url": url,
        "catalog_aliases": aliases,
        "default_endpoint": default_endpoint,
        "endpoints": endpoints,
    }


@pytest.fixture()
def temp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def load(payload: dict) -> Path:
        config_path = tmp_path / "gradio_tools.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setenv(GRADIO_TOOLS_CONFIG_ENV, str(config_path))
        reload_registry(config_path)
        return config_path

    yield load

    monkeypatch.delenv(GRADIO_TOOLS_CONFIG_ENV, raising=False)
    reload_registry()


def test_registry_resolves_tool_and_endpoint_aliases_to_distinct_endpoints(temp_registry) -> None:
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "stackreg",
                aliases=["StackReg", " stack registration "],
                default_endpoint="intra_stack",
                endpoints=[
                    _endpoint("intra_stack", aliases=["same-stack"]),
                    _endpoint("reference_align", aliases=["reference-stack"]),
                ],
            )
        ],
    }
    temp_registry(payload)

    tool_level = resolve_catalog_alias("STACKREG")
    endpoint_level = resolve_catalog_alias("reference-stack")

    assert list_tools() == ["stackreg"]
    assert tool_level is not None
    assert tool_level.endpoint_id == "intra_stack"
    assert alias_is_tool_level("stack registration", tool_level)
    assert endpoint_level is not None
    assert endpoint_level.endpoint_id == "reference_align"
    assert not alias_is_tool_level("reference-stack", endpoint_level)
    assert [tool.endpoint_id for tool in list_tool_endpoints("stackreg")] == [
        "intra_stack",
        "reference_align",
    ]


def test_registry_rejects_alias_collisions_after_normalization() -> None:
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "first_tool",
                aliases=[],
                endpoints=[_endpoint("segment", aliases=["Lungs-Segmentation"])],
            ),
            _tool(
                "second_tool",
                aliases=[],
                endpoints=[_endpoint("segment", aliases=[" lungs-segmentation "])],
            ),
        ],
    }

    with pytest.raises(RegistryValidationError, match="duplicate alias"):
        validate_config_payload(payload)


def test_pending_action_skips_unavailable_rank_and_uses_later_runnable_tool(temp_registry) -> None:
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "offline_demo",
                aliases=["offline-tool"],
                default_endpoint="run",
                endpoints=[_endpoint("run", aliases=["offline-tool"], demo_available=False)],
            ),
            _tool(
                "online_demo",
                aliases=["online-tool"],
                default_endpoint="run",
                endpoints=[_endpoint("run", aliases=["online-tool"])],
            ),
        ],
    }
    temp_registry(payload)
    session = Session(session_id="session-1")
    session.assets["asset-1"] = Asset(asset_id="asset-1", path="scan.tif")
    session.last_asset_ids = ["asset-1"]

    pending = _select_pending_action(
        session,
        choices=[
            {"name": "offline-tool", "why": "Best match but not runnable"},
            {"name": "online-tool", "why": "Second best with a configured endpoint"},
        ],
        effective_paths=["scan.tif"],
        request_text="segment this scan",
    )

    assert pending is not None
    assert pending.tool_name == "online_demo"
    assert pending.recommendation_rank == 2
    assert pending.endpoint_id == "run"
    assert session.pending_tool_approval == "online_demo"
    assert session.pending_recommendation_name == "online-tool"


def test_endpoint_specific_alias_waits_for_required_file_count(temp_registry) -> None:
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "pair_tool",
                aliases=["pair-tool"],
                default_endpoint="pair",
                endpoints=[
                    {
                        "id": "pair",
                        "display_name": "Pair",
                        "api_name": "/pair",
                        "catalog_aliases": ["pair-tool-two-files"],
                        "input_mapping": {
                            "parameters": [
                                {
                                    "name": "reference",
                                    "source": "session_file",
                                    "file_index": 0,
                                    "required": True,
                                },
                                {
                                    "name": "moving",
                                    "source": "session_file",
                                    "file_index": 1,
                                    "required": True,
                                },
                            ]
                        },
                    }
                ],
            )
        ],
    }
    temp_registry(payload)
    session = Session(session_id="session-1")
    session.assets["asset-1"] = Asset(asset_id="asset-1", path="reference.tif")
    session.last_asset_ids = ["asset-1"]

    pending = _select_pending_action(
        session,
        choices=[{"name": "pair-tool-two-files", "why": "Needs a pair"}],
        effective_paths=["reference.tif"],
        request_text="register two files",
    )

    assert pending is None
    assert session.pending_tool_approval is None

    pending = _select_pending_action(
        session,
        choices=[{"name": "pair-tool-two-files", "why": "Needs a pair"}],
        effective_paths=["reference.tif", "moving.tif"],
        request_text="register two files",
    )

    assert pending is not None
    assert pending.endpoint_id == "pair"
    assert session.pending_tool_params["image_paths"] == ["reference.tif", "moving.tif"]
