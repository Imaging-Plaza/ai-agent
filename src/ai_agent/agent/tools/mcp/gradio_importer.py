from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from ai_agent.agent.tools.mcp.registry import RegistryValidationError, validate_config_payload

_HF_SPACE_HOST_RE = re.compile(r"^(?P<slug>[a-z0-9][a-z0-9-]*)\.hf\.space$", re.IGNORECASE)
_SAFE_ID_RE = re.compile(r"[^a-z0-9]+")


def build_tool_config_from_space_url(space_url: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Fetch Gradio metadata and convert it into one registry tool entry."""
    base_url, tool_id, display_name = normalize_space_url(space_url)
    info = _fetch_json(f"{base_url}/gradio_api/info", timeout)
    mcp_schema = _fetch_json(f"{base_url}/gradio_api/mcp/schema", timeout)
    tool = _build_tool(base_url, tool_id, display_name, info, mcp_schema)

    # Validate the single generated entry before returning it to the API layer.
    validate_config_payload({"version": 1, "tools": [tool]})
    return tool


def normalize_space_url(value: str) -> tuple[str, str, str]:
    raw = (value or "").strip()
    if not raw:
        raise RegistryValidationError("Hugging Face Space URL is required")
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    host = (parsed.netloc or "").casefold()
    path_parts = [p for p in parsed.path.split("/") if p]

    direct = _HF_SPACE_HOST_RE.match(host)
    if direct:
        slug = direct.group("slug")
        return f"https://{host}", _slug_id(slug), _display_name_from_space_slug(slug)

    if host == "huggingface.co":
        if len(path_parts) >= 3 and path_parts[0] == "spaces":
            user, tool = path_parts[1], path_parts[2]
        elif len(path_parts) >= 2:
            user, tool = path_parts[0], path_parts[1]
        else:
            raise RegistryValidationError("Use a Space URL like user-tool.hf.space or huggingface.co/user/tool")
        slug = f"{user}-{tool}".casefold()
        return f"https://{slug}.hf.space", _slug_id(slug), _title_from_slug(tool)

    raise RegistryValidationError("Use a Space URL like user-tool.hf.space or huggingface.co/user/tool")


def _fetch_json(url: str, timeout: float) -> Any:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RegistryValidationError(f"Could not fetch {url}: {exc}") from exc
    except ValueError as exc:
        raise RegistryValidationError(f"{url} did not return JSON") from exc
    return data


def _build_tool(
    base_url: str,
    tool_id: str,
    display_name: str,
    info: Dict[str, Any],
    mcp_schema: Any,
) -> Dict[str, Any]:
    endpoints = _build_endpoints(info, mcp_schema)
    if not endpoints:
        raise RegistryValidationError("No callable Gradio endpoints were found for this Space")
    description = _first_text(
        info.get("description"),
        info.get("title"),
        _first_text(*(endpoint.get("description") for endpoint in endpoints)),
    )
    return {
        "id": tool_id,
        "display_name": _first_text(info.get("title"), display_name) or display_name,
        "description": description,
        "icon": "T",
        "enabled": True,
        "gradio_url": base_url,
        "catalog_aliases": _space_aliases(display_name, tool_id),
        "default_endpoint": endpoints[0]["id"],
        "endpoints": endpoints,
        "metadata": {
            "source": "hf_space_link",
            "info": _compact_metadata(info),
            "mcp_schema": _compact_metadata(mcp_schema),
        },
    }


def _build_endpoints(info: Dict[str, Any], mcp_schema: Any) -> List[Dict[str, Any]]:
    info_endpoints = _info_endpoints(info)
    schema_tools = _schema_tools(mcp_schema)
    if not info_endpoints and schema_tools:
        info_endpoints = {tool["name"]: {} for tool in schema_tools if tool.get("name")}

    endpoints: List[Dict[str, Any]] = []
    for index, (api_name, meta) in enumerate(info_endpoints.items()):
        if not api_name:
            continue
        schema_tool = _match_schema_tool(api_name, schema_tools)
        endpoint_id = _endpoint_id(api_name, index)
        display_name = _first_text(
            meta.get("name"),
            meta.get("display_name"),
            schema_tool.get("title") if schema_tool else None,
            schema_tool.get("name") if schema_tool else None,
            endpoint_id.replace("_", " ").title(),
        )
        description = _first_text(
            meta.get("description"),
            schema_tool.get("description") if schema_tool else None,
        )
        parameters = _parameters_for_endpoint(meta, schema_tool)
        endpoints.append(
            {
                "id": endpoint_id,
                "display_name": display_name,
                "description": description,
                "api_name": api_name if api_name.startswith("/") else f"/{api_name}",
                "enabled": True,
                "catalog_aliases": _unique_text([display_name, endpoint_id]),
                "supported_input_types": ["image", "file"],
                "input_mapping": {
                    "call_style": "keyword",
                    "parameters": parameters,
                },
                "output_mapping": {
                    "original": {"selector": "first", "materialize": True},
                    "preview": {"selector": "first", "materialize": True, "build_preview": True},
                },
                "approval": {
                    "required": True,
                    "message": f"Run {display_name} on your uploaded file?",
                },
                "demo": {"available": True},
                "metadata": {
                    "gradio_info": _compact_metadata(meta),
                    "mcp_tool": _compact_metadata(schema_tool or {}),
                },
            }
        )
    return endpoints


def _info_endpoints(info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = info.get("named_endpoints") or info.get("endpoints") or {}
    if isinstance(raw, dict):
        return {str(k): v if isinstance(v, dict) else {} for k, v in raw.items()}
    if isinstance(raw, list):
        found: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("api_name") or item.get("name") or item.get("id")
            if name:
                found[str(name)] = item
        return found
    return {}


def _schema_tools(schema: Any) -> List[Dict[str, Any]]:
    if isinstance(schema, list):
        return [tool for tool in schema if isinstance(tool, dict)]
    if not isinstance(schema, dict):
        return []
    tools = schema.get("tools")
    if isinstance(tools, list):
        return [tool for tool in tools if isinstance(tool, dict)]
    if isinstance(tools, dict):
        return [tool for tool in tools.values() if isinstance(tool, dict)]
    return []


def _match_schema_tool(api_name: str, tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    normalized = _slug_id(api_name)
    normalized_without_slash = _slug_id(api_name.lstrip("/"))
    for tool in tools:
        meta = tool.get("meta") if isinstance(tool.get("meta"), dict) else {}
        names = [tool.get("name"), tool.get("title"), tool.get("id"), meta.get("endpoint_name")]
        if any(_slug_id(str(name)) == normalized for name in names if name):
            return tool
        if any(_slug_id(str(name)) == normalized_without_slash for name in names if name):
            return tool
    if len(tools) == 1:
        return tools[0]
    return None


def _parameters_for_endpoint(meta: Dict[str, Any], schema_tool: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    schema_params = _schema_parameters(schema_tool or {})
    info_params = _info_parameters(meta)
    params = schema_params or info_params
    if not params:
        return [{"name": "file", "source": "session_file", "required": True, "as_gradio_file": True}]

    mapped: List[Dict[str, Any]] = []
    used_file = False
    for param in params:
        name = _slug_id(str(param.get("name") or f"param_{len(mapped) + 1}")) or f"param_{len(mapped) + 1}"
        required = bool(param.get("required", True))
        source = "param"
        as_file = False
        upload_to_gradio_path = False
        if not used_file and _looks_like_file_param(param):
            source = "session_file"
            as_file = _expects_gradio_file_payload(param)
            upload_to_gradio_path = not as_file
            used_file = True
        entry: Dict[str, Any] = {
            "name": name,
            "source": source,
            "required": required,
            "as_gradio_file": as_file,
            "metadata": {
                "description": param.get("description"),
                "type": param.get("type"),
                "format": param.get("format"),
                "choices": _extract_choices(param),
                "upload_to_gradio_path": upload_to_gradio_path,
            },
        }
        if "default" in param:
            entry["value"] = param.get("default")
        if source == "param":
            entry["param"] = name
        mapped.append(entry)
    return mapped


def _schema_parameters(schema_tool: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_schema = schema_tool.get("inputSchema") or schema_tool.get("input_schema") or {}
    properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
    if not isinstance(properties, dict):
        return []
    required = set(input_schema.get("required") or [])
    params = []
    for name, prop in properties.items():
        prop = prop if isinstance(prop, dict) else {}
        params.append(
            {
                "name": name,
                "required": name in required or bool(prop.get("required")),
                "type": prop.get("type"),
                "description": prop.get("description") or prop.get("title"),
                "format": prop.get("format"),
                "choices": _extract_choices(prop),
                "default": prop.get("default"),
            }
        )
    return params


def _info_parameters(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = meta.get("parameters") or meta.get("inputs") or []
    if not isinstance(raw, list):
        return []
    params = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("name") or item.get("parameter_name") or f"input_{index + 1}"
        component = item.get("component") or item.get("component_type") or item.get("type")
        has_default = bool(item.get("parameter_has_default", False))
        params.append(
            {
                "name": label,
                "required": not (bool(item.get("optional", False)) or has_default),
                "type": item.get("type"),
                "description": item.get("description") or item.get("label"),
                "component": component,
                "choices": _extract_choices(item),
                "default": item.get("parameter_default") if has_default else None,
            }
        )
    return params


def _extract_choices(value: Dict[str, Any]) -> List[Any]:
    for key in ("choices", "enum", "options"):
        raw = value.get(key)
        choices = _normalize_choices(raw)
        if choices:
            return choices
    type_info = value.get("type")
    if isinstance(type_info, dict):
        for key in ("choices", "enum", "options"):
            choices = _normalize_choices(type_info.get(key))
            if choices:
                return choices
    return []


def _normalize_choices(raw: Any) -> List[Any]:
    if not isinstance(raw, list):
        return []
    choices: List[Any] = []
    seen: set[str] = set()
    for item in raw:
        value = item
        if isinstance(item, (list, tuple)) and item:
            value = item[0]
        elif isinstance(item, dict):
            value = item.get("value", item.get("name", item.get("label")))
        if value in (None, ""):
            continue
        key = str(value)
        if key not in seen:
            seen.add(key)
            choices.append(value)
    return choices


def _looks_like_file_param(param: Dict[str, Any]) -> bool:
    text = " ".join(str(param.get(k) or "") for k in ("name", "type", "format", "component", "description")).casefold()
    return any(token in text for token in ("file", "image", "upload", "path", "dicom", "nifti", "tiff"))


def _expects_gradio_file_payload(param: Dict[str, Any]) -> bool:
    component = str(param.get("component") or "").casefold()
    if component in {"file", "uploadbutton", "image"}:
        return True
    text = " ".join(str(param.get(k) or "") for k in ("type", "format", "description")).casefold()
    return "filedata" in text or "gradio file input" in text


def _endpoint_id(api_name: str, index: int) -> str:
    value = api_name.strip().lstrip("/")
    return _slug_id(value) or f"endpoint_{index + 1}"


def _slug_id(value: str) -> str:
    return _SAFE_ID_RE.sub("_", (value or "").strip().casefold()).strip("_")


def _title_from_slug(value: str) -> str:
    return " ".join(part.capitalize() for part in _SAFE_ID_RE.split(value or "") if part)


def _display_name_from_space_slug(value: str) -> str:
    parts = [part for part in _SAFE_ID_RE.split(value or "") if part]
    if len(parts) > 2:
        parts = parts[1:]
    if len(parts) > 1 and parts[-1].casefold() in {"app", "space", "demo"}:
        parts = parts[:-1]
    return " ".join(part.capitalize() for part in parts)


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _unique_text(values: Iterable[Optional[str]]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        if not value:
            continue
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _space_aliases(display_name: str, tool_id: str) -> List[str]:
    aliases: List[Optional[str]] = [display_name, tool_id]
    words = [part for part in _SAFE_ID_RE.split(display_name or "") if part]
    if len(words) > 1 and words[-1].casefold() in {"app", "space", "demo"}:
        short = " ".join(words[:-1])
        aliases.extend([short, _slug_id(short)])
    return _unique_text(aliases)


def _compact_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _compact_metadata(v) for k, v in value.items() if k not in {"dependencies"}}
    if isinstance(value, list):
        return [_compact_metadata(v) for v in value[:50]]
    return value
