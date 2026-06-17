from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_agent.agent.tools.mcp import reload_registry, resolve_catalog_alias
from ai_agent.agent.tools.mcp.registry import GRADIO_TOOLS_CONFIG_ENV
from ai_agent.services.chat import _select_endpoint_for_choice


def _write_config(path: Path) -> None:
    payload = {
        "version": 1,
        "tools": [
            {
                "id": "pystackreg",
                "display_name": "PyStackReg",
                "enabled": True,
                "gradio_url": "https://example.com/",
                "catalog_aliases": ["pystackreg"],
                "default_endpoint": "intra_stack_align",
                "endpoints": [
                    {
                        "id": "intra_stack_align",
                        "display_name": "Intra-stack Align",
                        "description": "Align frames within one uploaded stack.",
                        "api_name": "/intra_stack_align",
                        "catalog_aliases": ["same-stack-registration"],
                        "input_mapping": {
                            "parameters": [
                                {
                                    "name": "stack_file",
                                    "source": "session_file",
                                    "file_index": 0,
                                    "required": True,
                                }
                            ]
                        },
                    },
                    {
                        "id": "reference_align",
                        "display_name": "Reference Align",
                        "description": "Register a moving stack to a separate reference stack.",
                        "api_name": "/reference_align",
                        "catalog_aliases": ["reference-based-registration"],
                        "input_mapping": {
                            "parameters": [
                                {
                                    "name": "reference_stack_file",
                                    "source": "session_file",
                                    "file_index": 0,
                                    "required": True,
                                },
                                {
                                    "name": "moving_stack_file",
                                    "source": "session_file",
                                    "file_index": 1,
                                    "required": True,
                                },
                            ]
                        },
                    },
                    {
                        "id": "frame_to_frame_align",
                        "display_name": "Frame-to-frame Align",
                        "description": "Align a moving frame to a reference frame.",
                        "api_name": "/frame_to_frame_align",
                        "catalog_aliases": ["frame-to-frame-registration"],
                        "input_mapping": {
                            "parameters": [
                                {
                                    "name": "stack_file",
                                    "source": "session_file",
                                    "file_index": 0,
                                    "required": True,
                                }
                            ]
                        },
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def pystackreg_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "gradio_tools.json"
    _write_config(config_path)
    monkeypatch.setenv(GRADIO_TOOLS_CONFIG_ENV, str(config_path))
    reload_registry(config_path)
    yield
    monkeypatch.delenv(GRADIO_TOOLS_CONFIG_ENV, raising=False)
    reload_registry()


def _select(request_text: str, file_count: int = 2, alias: str = "pystackreg") -> str:
    tool = resolve_catalog_alias(alias)
    assert tool is not None
    selected = _select_endpoint_for_choice(
        tool_config=tool,
        alias=alias,
        choice={"name": alias, "why": ""},
        request_text=request_text,
        file_count=file_count,
    )
    assert selected.tool.endpoint_id is not None
    return selected.tool.endpoint_id


def test_generic_pystackreg_selects_reference_endpoint(pystackreg_registry) -> None:
    endpoint_id = _select(
        "Register this moving stack to the separate reference stack.",
        file_count=2,
    )

    assert endpoint_id == "reference_align"


def test_generic_pystackreg_selects_frame_endpoint(pystackreg_registry) -> None:
    endpoint_id = _select(
        "Align one moving frame to a reference frame in this TIFF stack.",
        file_count=1,
    )

    assert endpoint_id == "frame_to_frame_align"


def test_generic_pystackreg_keeps_intra_stack_for_single_stack(
    pystackreg_registry,
) -> None:
    endpoint_id = _select(
        "Correct drift within the same uploaded stack.",
        file_count=1,
    )

    assert endpoint_id == "intra_stack_align"


def test_endpoint_specific_alias_bypasses_generic_selection(
    pystackreg_registry,
) -> None:
    endpoint_id = _select(
        "This text mentions frame, but the alias is explicit.",
        file_count=2,
        alias="reference-based-registration",
    )

    assert endpoint_id == "reference_align"
