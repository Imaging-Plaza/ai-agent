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
    resolve_runnable_url,
    validate_config_payload,
)
from ai_agent.agent.tools.mcp.registry import GRADIO_TOOLS_CONFIG_ENV
from ai_agent.agent.tools.mcp.gradio_importer import build_tool_config_from_space_url, normalize_space_url
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


def test_registry_resolves_huggingface_runnable_url_variants(temp_registry) -> None:
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "lungs_space",
                aliases=[],
                url="https://qchapp-3d-lungs-segmentation.hf.space/",
                endpoints=[_endpoint("segment", aliases=[])],
            )
        ],
    }
    temp_registry(payload)

    direct = resolve_runnable_url("https://qchapp-3d-lungs-segmentation.hf.space/")
    hf_repo = resolve_runnable_url("https://huggingface.co/qchapp/3d-lungs-segmentation")
    hf_spaces = resolve_runnable_url("https://huggingface.co/spaces/qchapp/3d-lungs-segmentation")

    assert direct is not None
    assert direct.name == "lungs_space"
    assert hf_repo is not None
    assert hf_repo.name == "lungs_space"
    assert hf_spaces is not None
    assert hf_spaces.name == "lungs_space"


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


def test_pending_action_exposes_runtime_parameters(temp_registry) -> None:
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "runtime_tool",
                aliases=["runtime-tool"],
                default_endpoint="run",
                endpoints=[
                    {
                        "id": "run",
                        "display_name": "Run",
                        "api_name": "/run",
                        "catalog_aliases": ["runtime-tool"],
                        "input_mapping": {
                            "parameters": [
                                {
                                    "name": "image",
                                    "source": "session_file",
                                    "file_index": 0,
                                    "required": True,
                                },
                                {
                                    "name": "threshold",
                                    "source": "param",
                                    "param": "threshold",
                                    "required": True,
                                    "value": 0.5,
                                    "metadata": {"description": "Mask threshold", "choices": [0.25, 0.5, 0.75]},
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
    session.assets["asset-1"] = Asset(asset_id="asset-1", path="scan.tif")
    session.last_asset_ids = ["asset-1"]

    pending = _select_pending_action(
        session,
        choices=[{"name": "runtime-tool", "why": "Needs a threshold"}],
        effective_paths=["scan.tif"],
        request_text="segment with threshold",
    )

    assert pending is not None
    assert pending.runtime_parameters[0].name == "threshold"
    assert pending.runtime_parameters[0].description == "Mask threshold"
    assert pending.runtime_parameters[0].default == 0.5
    assert pending.runtime_parameters[0].choices == [0.25, 0.5, 0.75]


def test_pending_action_exposes_selectable_endpoint_options(temp_registry) -> None:
    payload = {
        "version": 1,
        "tools": [
            _tool(
                "multi_endpoint_tool",
                aliases=["multi-endpoint-tool"],
                default_endpoint="default_run",
                endpoints=[
                    {
                        "id": "default_run",
                        "display_name": "Default Run",
                        "api_name": "/default_run",
                        "catalog_aliases": ["multi-endpoint-tool"],
                        "input_mapping": {
                            "parameters": [
                                {"name": "image", "source": "session_file", "required": True},
                                {
                                    "name": "mode",
                                    "source": "param",
                                    "param": "mode",
                                    "required": False,
                                    "value": "fast",
                                    "metadata": {"choices": ["fast", "accurate"]},
                                },
                            ]
                        },
                    },
                    {
                        "id": "alternate_run",
                        "display_name": "Alternate Run",
                        "description": "Use this endpoint for alternate processing.",
                        "api_name": "/alternate_run",
                        "catalog_aliases": ["alternate-run"],
                        "input_mapping": {
                            "parameters": [
                                {"name": "image", "source": "session_file", "required": True},
                                {
                                    "name": "quality",
                                    "source": "param",
                                    "param": "quality",
                                    "required": True,
                                    "value": 2,
                                },
                            ]
                        },
                    },
                ],
            )
        ],
    }
    temp_registry(payload)
    session = Session(session_id="session-1")

    pending = _select_pending_action(
        session,
        choices=[{"name": "multi-endpoint-tool", "why": "Run the tool"}],
        effective_paths=["scan.tif"],
        request_text="run the tool",
    )

    assert pending is not None
    assert pending.endpoint_id == "default_run"
    assert [option.endpoint_id for option in pending.endpoint_options] == [
        "default_run",
        "alternate_run",
    ]
    assert pending.endpoint_options[0].runtime_parameters[0].name == "mode"
    assert pending.endpoint_options[0].runtime_parameters[0].choices == ["fast", "accurate"]
    assert pending.endpoint_options[1].runtime_parameters[0].name == "quality"


def test_normalize_space_url_accepts_direct_and_huggingface_forms() -> None:
    assert normalize_space_url("qchapp-3d-lungs-segmentation.hf.space")[0] == (
        "https://qchapp-3d-lungs-segmentation.hf.space"
    )
    assert normalize_space_url("https://huggingface.co/qchapp/3d-lungs-segmentation")[0] == (
        "https://qchapp-3d-lungs-segmentation.hf.space"
    )
    assert normalize_space_url("https://huggingface.co/spaces/qchapp/3d-lungs-segmentation")[0] == (
        "https://qchapp-3d-lungs-segmentation.hf.space"
    )


def test_build_tool_config_from_space_url_fetches_info_and_mcp_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    info = {
        "title": "3D Lungs Segmentation",
        "description": "Segments lungs from volumes.",
        "named_endpoints": {
            "/predict": {
                "parameters": [
                    {"label": "Scan file", "type": "file", "description": "Input volume"},
                    {"label": "Threshold", "type": "number", "description": "Mask threshold"},
                ]
            }
        },
    }
    schema = {
        "tools": [
            {
                "name": "predict",
                "description": "Run segmentation",
                "inputSchema": {
                    "type": "object",
                    "required": ["scan_file"],
                    "properties": {
                        "scan_file": {"type": "string", "description": "Input volume"},
                        "threshold": {"type": "number", "description": "Mask threshold"},
                        "mode": {
                            "type": "string",
                            "description": "Transform mode",
                            "enum": ["RIGID_BODY", "AFFINE"],
                            "default": "RIGID_BODY",
                        },
                    },
                },
            }
        ]
    }

    def fake_get(url: str, timeout: float):
        if url.endswith("/gradio_api/info"):
            return FakeResponse(info)
        if url.endswith("/gradio_api/mcp/schema"):
            return FakeResponse(schema)
        raise AssertionError(url)

    monkeypatch.setattr("ai_agent.agent.tools.mcp.gradio_importer.requests.get", fake_get)

    tool = build_tool_config_from_space_url("qchapp-3d-lungs-segmentation.hf.space")

    assert tool["id"] == "qchapp_3d_lungs_segmentation"
    assert tool["gradio_url"] == "https://qchapp-3d-lungs-segmentation.hf.space"
    endpoint = tool["endpoints"][0]
    assert endpoint["api_name"] == "/predict"
    assert endpoint["description"] == "Run segmentation"
    assert endpoint["input_mapping"]["parameters"][0]["source"] == "session_file"
    assert endpoint["input_mapping"]["parameters"][0]["as_gradio_file"] is False
    assert endpoint["input_mapping"]["parameters"][0]["metadata"]["upload_to_gradio_path"] is True
    assert endpoint["input_mapping"]["parameters"][1]["source"] == "param"
    assert endpoint["input_mapping"]["parameters"][2]["metadata"]["choices"] == [
        "RIGID_BODY",
        "AFFINE",
    ]
    assert endpoint["input_mapping"]["parameters"][2]["value"] == "RIGID_BODY"
    assert endpoint["contracts"]["inputs"][0]["name"] == "scan_file"
    assert endpoint["contracts"]["outputs"][0]["semantic_roles"] == [
        "segment",
        "mask",
        "segmentation",
    ]


def test_build_tool_config_from_space_url_accepts_top_level_mcp_tool_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload: dict | list) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict | list:
            return self.payload

    info = {
        "named_endpoints": {
            "/segment_lungs_3d_tiff": {
                "parameters": [
                    {
                        "label": "Input 3D TIF/TIFF volume",
                        "parameter_name": "volume_tiff",
                        "component": "File",
                    }
                ]
            }
        }
    }
    schema = [
        {
            "name": "3d_lungs_segmentation_segment_lungs_3d_tiff",
            "description": "Recommended MCP tool for agentic clients.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "volume_tiff": {
                        "type": "string",
                        "description": "Uploaded `.tif` or `.tiff` 3D volume",
                        "format": "Gradio File Input - a http or https url to a file",
                    }
                },
            },
            "meta": {"endpoint_name": "segment_lungs_3d_tiff"},
        }
    ]

    def fake_get(url: str, timeout: float):
        if url.endswith("/gradio_api/info"):
            return FakeResponse(info)
        if url.endswith("/gradio_api/mcp/schema"):
            return FakeResponse(schema)
        raise AssertionError(url)

    monkeypatch.setattr("ai_agent.agent.tools.mcp.gradio_importer.requests.get", fake_get)

    tool = build_tool_config_from_space_url("qchapp-3d-lungs-segmentation.hf.space")
    endpoint = tool["endpoints"][0]

    assert endpoint["id"] == "segment_lungs_3d_tiff"
    assert endpoint["description"] == "Recommended MCP tool for agentic clients."
    assert endpoint["input_mapping"]["parameters"][0]["name"] == "volume_tiff"
    assert endpoint["input_mapping"]["parameters"][0]["source"] == "session_file"
    assert endpoint["input_mapping"]["parameters"][0]["required"] is True
    assert endpoint["input_mapping"]["parameters"][0]["as_gradio_file"] is True
    assert endpoint["input_mapping"]["parameters"][0]["metadata"]["upload_to_gradio_path"] is False
    assert endpoint["contracts"]["inputs"][0]["formats"] == ["tif", "tiff"]
    assert endpoint["contracts"]["outputs"][0]["artifact_type"] == "mask.volume.3d"


def test_build_tool_config_from_space_url_infers_pystackreg_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __init__(self, payload: dict | list) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict | list:
            return self.payload

    info = {
        "named_endpoints": {
            "/align_stack_to_reference": {
                "description": "Align every frame in a TIFF stack to a chosen reference frame.",
                "parameters": [
                    {
                        "label": "1st",
                        "parameter_name": "stack_file",
                        "parameter_has_default": False,
                        "parameter_default": None,
                        "type": {"type": "string"},
                    },
                    {
                        "label": "2nd",
                        "parameter_name": "reference_index",
                        "parameter_has_default": True,
                        "parameter_default": 0,
                        "type": {"type": "integer"},
                    },
                ],
                "returns": [{"label": "Aligned output TIFF file", "component": "Api"}],
            }
        }
    }
    schema = [
        {
            "name": "pystackreg_app__mcp_align_stack_to_reference",
            "description": "Align every frame in stack_file. Returns: The aligned output TIFF file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "stack_file": {
                        "type": "string",
                        "description": "Path or HTTP/HTTPS URL to the input TIFF stack.",
                    },
                    "reference_index": {
                        "type": "integer",
                        "description": "Zero-based index of the reference frame.",
                        "default": 0,
                    },
                },
            },
            "meta": {"endpoint_name": "align_stack_to_reference"},
        }
    ]

    def fake_get(url: str, timeout: float):
        if url.endswith("/gradio_api/info"):
            return FakeResponse(info)
        if url.endswith("/gradio_api/mcp/schema"):
            return FakeResponse(schema)
        raise AssertionError(url)

    monkeypatch.setattr("ai_agent.agent.tools.mcp.gradio_importer.requests.get", fake_get)

    tool = build_tool_config_from_space_url("qchapp-pystackreg-app.hf.space")
    endpoint = tool["endpoints"][0]

    assert endpoint["input_mapping"]["parameters"][0]["name"] == "stack_file"
    assert endpoint["input_mapping"]["parameters"][0]["required"] is True
    assert endpoint["input_mapping"]["parameters"][0]["source"] == "session_file"
    assert endpoint["contracts"]["inputs"][0]["artifact_type"] == "image.volume.3d"
    assert endpoint["contracts"]["outputs"][0]["name"] == "aligned_stack"
    assert endpoint["contracts"]["outputs"][0]["semantic_roles"] == [
        "align",
        "aligned",
        "registered",
    ]
