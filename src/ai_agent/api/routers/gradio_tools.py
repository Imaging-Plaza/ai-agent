from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ai_agent.agent.tools.mcp import (
    GRADIO_TOOLS_CONFIG_ENV,
    RegistryValidationError,
    active_config_json,
    active_config_path,
    reload_registry,
    save_config_payload,
    validate_config_payload,
)
from ai_agent.agent.tools.mcp.gradio_importer import build_tool_config_from_space_url
from ai_agent.api.deps import require_auth

log = logging.getLogger("api.routers.gradio_tools")

router = APIRouter(prefix="/api/gradio-tools", tags=["gradio-tools"], dependencies=[Depends(require_auth)])


class ConfigEnvelope(BaseModel):
    config: Dict[str, Any]


class ConfigReadResponse(BaseModel):
    config: Dict[str, Any]
    path: str
    override_env: str = GRADIO_TOOLS_CONFIG_ENV


class ValidationResponse(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)


class SaveResponse(BaseModel):
    ok: bool
    path: str
    reloaded: bool
    restart_required: bool = False
    errors: list[str] = Field(default_factory=list)


class ImportLinkRequest(BaseModel):
    url: str


@router.get("", response_model=ConfigReadResponse)
def read_config() -> ConfigReadResponse:
    try:
        return ConfigReadResponse(config=active_config_json(), path=str(active_config_path()))
    except RegistryValidationError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/validate", response_model=ValidationResponse)
def validate_config(body: ConfigEnvelope) -> ValidationResponse:
    try:
        validate_config_payload(body.config)
        return ValidationResponse(ok=True)
    except RegistryValidationError as exc:
        return ValidationResponse(ok=False, errors=[str(exc)])


@router.post("/save", response_model=SaveResponse)
def save_config(body: ConfigEnvelope) -> SaveResponse:
    try:
        path = save_config_payload(body.config)
    except RegistryValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        reload_registry(path)
    except RegistryValidationError as exc:
        log.exception("Saved Gradio tools config but reload failed")
        return SaveResponse(ok=False, path=str(path), reloaded=False, restart_required=True, errors=[str(exc)])
    return SaveResponse(ok=True, path=str(path), reloaded=True)


@router.post("/import-link", response_model=SaveResponse)
def import_link(body: ImportLinkRequest) -> SaveResponse:
    try:
        tool = build_tool_config_from_space_url(body.url)
        config = active_config_json()
        existing_tools = list(config.get("tools") or [])
        if any(item.get("id") == tool["id"] for item in existing_tools if isinstance(item, dict)):
            raise RegistryValidationError(f"A tool with id {tool['id']!r} already exists")
        config["version"] = config.get("version") or 1
        config["tools"] = [*existing_tools, tool]
        path = save_config_payload(config)
    except RegistryValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        reload_registry(path)
    except RegistryValidationError as exc:
        log.exception("Saved imported Gradio tool but reload failed")
        return SaveResponse(ok=False, path=str(path), reloaded=False, restart_required=True, errors=[str(exc)])
    return SaveResponse(ok=True, path=str(path), reloaded=True)


@router.post("/reload", response_model=SaveResponse)
def reload_config() -> SaveResponse:
    path = active_config_path()
    try:
        reload_registry(path)
    except RegistryValidationError as exc:
        return SaveResponse(ok=False, path=str(path), reloaded=False, restart_required=True, errors=[str(exc)])
    return SaveResponse(ok=True, path=str(path), reloaded=True)
