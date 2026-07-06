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


def test_build_inputs_uses_param_default_when_not_supplied(
    executor_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    monkeypatch.setattr(gradio_executor, "handle_file", lambda value: {"path": value})
    tool = get_tool("demo_tool", "process")
    assert tool is not None
    threshold = next(p for p in tool.endpoint.input_mapping.parameters if p.name == "threshold")
    threshold.value = 0.5

    args, _ = gradio_executor._build_inputs(
        tool,
        GenericGradioInput(
            tool_id="demo_tool",
            endpoint_id="process",
            image_path=str(image),
            description="align carefully",
        ),
    )

    assert args[3] == 0.5


def test_build_inputs_omits_blank_optional_keyword_params_but_keeps_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "version": 1,
        "tools": [
            {
                "id": "keyword_tool",
                "display_name": "Keyword Tool",
                "enabled": True,
                "gradio_url": "https://example.com/",
                "default_endpoint": "run",
                "endpoints": [
                    {
                        "id": "run",
                        "display_name": "Run",
                        "api_name": "/run",
                        "input_mapping": {
                            "call_style": "keyword",
                            "parameters": [
                                {
                                    "name": "stack_file",
                                    "source": "session_file",
                                    "required": True,
                                    "as_gradio_file": True,
                                },
                                {
                                    "name": "reference_index",
                                    "source": "param",
                                    "param": "reference_index",
                                    "required": False,
                                },
                                {
                                    "name": "mode",
                                    "source": "param",
                                    "param": "mode",
                                    "required": False,
                                    "value": None,
                                },
                                {
                                    "name": "external_reference_file",
                                    "source": "param",
                                    "param": "external_reference_file",
                                    "required": False,
                                    "value": None,
                                },
                            ]
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
    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    monkeypatch.setattr(gradio_executor, "handle_file", lambda value: {"path": value})
    tool = get_tool("keyword_tool", "run")
    assert tool is not None

    _, kwargs = gradio_executor._build_inputs(
        tool,
        GenericGradioInput(
            tool_id="keyword_tool",
            endpoint_id="run",
            image_path=str(image),
            params={"reference_index": 0},
        ),
    )

    assert kwargs == {"stack_file": {"path": str(image)}, "reference_index": 0}
    monkeypatch.delenv(GRADIO_TOOLS_CONFIG_ENV, raising=False)
    reload_registry()


def test_build_inputs_uploads_path_string_inputs_to_gradio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "version": 1,
        "tools": [
            {
                "id": "path_tool",
                "display_name": "Path Tool",
                "enabled": True,
                "gradio_url": "https://example.com/",
                "default_endpoint": "run",
                "endpoints": [
                    {
                        "id": "run",
                        "display_name": "Run",
                        "api_name": "/run",
                        "input_mapping": {
                            "call_style": "keyword",
                            "parameters": [
                                {
                                    "name": "stack_file",
                                    "source": "session_file",
                                    "required": True,
                                    "as_gradio_file": False,
                                    "metadata": {"upload_to_gradio_path": True},
                                },
                                {
                                    "name": "reference_index",
                                    "source": "param",
                                    "param": "reference_index",
                                    "required": False,
                                },
                            ]
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
    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    tool = get_tool("path_tool", "run")
    assert tool is not None
    captured: dict[str, object] = {}

    class FakeClient:
        upload_url = "https://example.com/upload"
        src = "https://example.com"
        headers = {"x-test": "1"}
        cookies = None
        ssl_verify = True
        httpx_kwargs = {"timeout": 5}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return ["/tmp/gradio/uploaded_scan.tif"]

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["files_name"] = kwargs["files"][0][1][0]
        return FakeResponse()

    monkeypatch.setattr(gradio_executor.requests, "post", fake_post)

    _, kwargs = gradio_executor._build_inputs(
        tool,
        GenericGradioInput(
            tool_id="path_tool",
            endpoint_id="run",
            image_path=str(image),
            params={"reference_index": 0},
        ),
        client=FakeClient(),
    )

    assert kwargs == {
        "stack_file": "https://example.com/gradio_api/file=/tmp/gradio/uploaded_scan.tif",
        "reference_index": 0,
    }
    assert captured == {"url": "https://example.com/upload", "files_name": "scan.tif"}
    monkeypatch.delenv(GRADIO_TOOLS_CONFIG_ENV, raising=False)
    reload_registry()


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


def test_execute_gradio_endpoint_runs_configured_preflight_calls(
    executor_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def predict(self, *args, api_name=None, **kwargs):
            self.calls.append((args, kwargs, api_name))
            if api_name == "/prepare":
                return {"maximum": 4, "value": 0, "__type__": "update"}
            return {
                "ok": True,
                "files": {
                    "origin": {"path": "/remote/origin.tif"},
                    "preview": {"path": "/remote/preview.png"},
                },
            }

    fake_client = FakeClient()
    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    tool = get_tool("demo_tool", "process")
    assert tool is not None
    tool.endpoint.metadata["preflight_calls"] = [
        {"api_name": "/prepare", "arg_indexes": [0]}
    ]
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
        )
    )

    assert result.success is True
    assert fake_client.calls[0] == (({"path": str(image)},), {}, "/prepare")
    assert fake_client.calls[1][2] == "/process"


def test_execute_gradio_endpoint_keeps_successful_undownloadable_string_response(
    executor_registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def predict(self, *args, api_name=None, **kwargs):
            return "/tmp/psr_cache/work/aligned.tif"

    image = tmp_path / "scan.tif"
    image.write_bytes(b"tif")
    monkeypatch.setattr(gradio_executor, "handle_file", lambda value: {"path": value})
    monkeypatch.setattr(gradio_executor, "_make_client", lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(gradio_executor, "_materialize_any", lambda *args, **kwargs: None)

    result = gradio_executor.execute_gradio_endpoint(
        GenericGradioInput(
            tool_id="demo_tool",
            endpoint_id="process",
            image_path=str(image),
            params={"threshold": 0.3},
        )
    )

    assert result.success is True
    assert result.result_origin is None
    assert result.result_preview is None
    assert "aligned.tif" in (result.metadata_text or "")
    assert "server-local path" in (result.notes or "")
    assert "allowed_paths" in (result.notes or "")
