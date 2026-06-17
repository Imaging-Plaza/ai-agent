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

from ai_agent.agent.tools.mcp import reload_registry
from ai_agent.agent.tools.mcp.registry import (
    GRADIO_TOOLS_CONFIG_ENV,
    GenericGradioInput,
    get_tool,
)
from ai_agent.agent.tools.mcp import gradio_executor


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def executor_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = {
        "version": 1,
        "tools": [
            {
                "id": "demo_tool",
                "display_name": "Demo Tool",
                "enabled": True,
                "gradio_url": "https://example.com/",
                "catalog_aliases": ["demo-tool"],
                "default_endpoint": "process",
                "endpoints": [
                    {
                        "id": "process",
                        "display_name": "Process",
                        "api_name": "/process",
                        "catalog_aliases": ["demo-process"],
                        "input_mapping": {
                            "call_style": "positional",
                            "parameters": [
                                {
                                    "name": "primary_file",
                                    "source": "session_file",
                                    "file_index": 0,
                                    "required": True,
                                    "as_gradio_file": True,
                                },
                                {
                                    "name": "optional_reference",
                                    "source": "session_file",
                                    "file_index": 1,
                                    "required": False,
                                    "as_gradio_file": True,
                                },
                                {
                                    "name": "mode",
                                    "source": "literal",
                                    "value": "rigid",
                                },
                                {
                                    "name": "threshold",
                                    "source": "param",
                                    "param": "threshold",
                                    "required": True,
                                },
                                {
                                    "name": "notes",
                                    "source": "description",
                                    "required": False,
                                },
                            ],
                        },
                        "output_mapping": {
                            "original": {"selector": "$.files.origin"},
                            "preview": {"selector": "$.files.preview"},
                            "metadata": "$.meta",
                            "notes": "$.notes",
                            "success": "$.ok",
                            "error": "$.error",
                            "compute_time": "$.seconds",
                        },
                    }
                ],
            }
        ],
    }
    config_path = tmp_path / "gradio_tools.json"
    _write_config(config_path, payload)
    monkeypatch.setenv(GRADIO_TOOLS_CONFIG_ENV, str(config_path))
    reload_registry(config_path)
    yield
    monkeypatch.delenv(GRADIO_TOOLS_CONFIG_ENV, raising=False)
    reload_registry()


def test_build_inputs_supports_files_literals_params_and_description(
    executor_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    monkeypatch.setattr(gradio_executor, "handle_file", lambda value: {"path": value})
    tool = get_tool("demo_tool", "process")
    assert tool is not None

    args, kwargs = gradio_executor._build_inputs(
        tool,
        GenericGradioInput(
            tool_id="demo_tool",
            endpoint_id="process",
            image_path=str(image),
            description="align carefully",
            params={"threshold": 0.75},
        ),
    )

    assert kwargs == {}
    assert args == [
        {"path": str(image)},
        None,
        "rigid",
        0.75,
        "align carefully",
    ]


def test_build_inputs_reports_missing_required_param(executor_registry, tmp_path: Path) -> None:
    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    tool = get_tool("demo_tool", "process")
    assert tool is not None

    with pytest.raises(ValueError, match="threshold"):
        gradio_executor._build_inputs(
            tool,
            GenericGradioInput(
                tool_id="demo_tool",
                endpoint_id="process",
                image_path=str(image),
            ),
        )


@pytest.mark.parametrize(
    "raw, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("0", False),
        ("1", True),
    ],
)
def test_extract_bool_understands_common_gradio_string_values(raw, expected) -> None:
    assert gradio_executor._extract_bool({"ok": raw}, "$.ok") is expected


def test_execute_gradio_endpoint_maps_response_fields_without_network(
    executor_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, *args, api_name=None, **kwargs):
            self.calls.append((args, kwargs, api_name))
            return {
                "ok": True,
                "files": {
                    "origin": {"path": "/remote/origin.tif"},
                    "preview": {"path": "/remote/preview.png"},
                },
                "meta": "size=10x10",
                "notes": "done",
                "seconds": "1.25",
            }

    fake_client = FakeClient()
    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    monkeypatch.setattr(gradio_executor, "handle_file", lambda value: {"path": value})
    monkeypatch.setattr(gradio_executor, "_make_client", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(
        gradio_executor,
        "_materialize_any",
        lambda selected, client, token, max_bytes: f"/local/{Path(selected['path']).name}",
    )

    result = gradio_executor.execute_gradio_endpoint(
        GenericGradioInput(
            tool_id="demo_tool",
            endpoint_id="process",
            image_path=str(image),
            params={"threshold": 0.3},
            description="optional input only",
        )
    )

    assert result.success is True
    assert result.result_origin == "/local/origin.tif"
    assert result.result_preview == "/local/preview.png"
    assert result.result_path == "/local/preview.png"
    assert result.metadata_text == "size=10x10"
    assert result.notes == "done"
    assert result.compute_time_seconds == 1.25
    assert result.api_name == "/process"
    assert fake_client.calls[0][2] == "/process"
