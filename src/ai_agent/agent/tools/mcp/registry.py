from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator, model_validator

from ai_agent.agent.tools.mcp.base import BaseToolOutput

GRADIO_TOOLS_CONFIG_ENV = "AI_AGENT_GRADIO_TOOLS_CONFIG"
DEFAULT_CONFIG_PACKAGE = "ai_agent.config"
DEFAULT_CONFIG_NAME = "gradio_tools.json"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_FIELD_RE = re.compile(r"(token|secret|password|key)", re.IGNORECASE)


class RegistryValidationError(ValueError):
    """Raised when a Gradio tool config cannot produce a safe registry."""


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_env: Optional[str] = None
    token_envs: List[str] = Field(default_factory=list)
    required: bool = False

    @field_validator("token_env")
    @classmethod
    def _valid_token_env(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not _ENV_NAME_RE.match(value):
            raise ValueError(f"auth.token_env must be an environment variable name, got {value!r}")
        if _looks_like_secret(value):
            raise ValueError("auth.token_env must name an environment variable, not contain a secret value")
        return value

    @field_validator("token_envs")
    @classmethod
    def _valid_token_envs(cls, values: List[str]) -> List[str]:
        for value in values:
            if not _ENV_NAME_RE.match(value):
                raise ValueError(f"auth.token_envs entries must be environment variable names, got {value!r}")
            if _looks_like_secret(value):
                raise ValueError("auth.token_envs entries must name environment variables, not contain secret values")
        return values

    def candidate_envs(self) -> List[str]:
        names: List[str] = []
        if self.token_env:
            names.append(self.token_env)
        names.extend(self.token_envs)
        return list(dict.fromkeys(names))


class InputParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source: Literal["session_file", "image_path", "description", "literal", "param"] = "session_file"
    required: bool = True
    value: Any = None
    param: Optional[str] = None
    file_index: int = 0
    as_gradio_file: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input parameter name cannot be empty")
        return value.strip()

    @field_validator("file_index")
    @classmethod
    def _valid_file_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("file_index must be zero or greater")
        return value


class InputMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_style: Literal["keyword", "positional"] = "keyword"
    parameters: List[InputParameter] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_parameters(self) -> "InputMapping":
        if not self.parameters:
            raise ValueError("input_mapping.parameters must contain at least one parameter")
        return self


class OutputSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selector: str = "first"
    materialize: bool = True
    build_preview: bool = False


class OutputMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original: OutputSelector = Field(default_factory=lambda: OutputSelector(selector="first", materialize=True))
    preview: OutputSelector = Field(default_factory=lambda: OutputSelector(selector="first", materialize=True, build_preview=True))
    metadata: Optional[str] = None
    notes: Optional[str] = None
    success: Optional[str] = None
    error: Optional[str] = None
    compute_time: Optional[str] = None


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = True
    title: Optional[str] = None
    message: Optional[str] = None


class DemoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = True


class GradioEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    description: Optional[str] = None
    api_name: Optional[str] = None
    enabled: bool = True
    catalog_aliases: List[str] = Field(default_factory=list)
    supported_input_types: List[str] = Field(default_factory=lambda: ["image", "file"])
    input_mapping: InputMapping
    output_mapping: OutputMapping = Field(default_factory=OutputMapping)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)
    timeout_seconds: Optional[float] = None
    max_download_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "display_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field cannot be empty")
        return value.strip()

    @field_validator("api_name")
    @classmethod
    def _api_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("api_name cannot be empty when present")
        return value.strip() if value else value

    @field_validator("catalog_aliases")
    @classmethod
    def _aliases(cls, values: List[str]) -> List[str]:
        cleaned = [_normalize_alias(v) for v in values]
        if any(not v for v in cleaned):
            raise ValueError("catalog_aliases cannot contain empty values")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("catalog_aliases contains duplicate values")
        return cleaned

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        return value

    @field_validator("max_download_bytes")
    @classmethod
    def _download_limit(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value <= 0:
            raise ValueError("max_download_bytes must be greater than zero")
        return value

    @model_validator(mode="after")
    def _enabled_requires_api(self) -> "GradioEndpointConfig":
        if self.enabled and not self.api_name:
            raise ValueError(f"enabled endpoint {self.id!r} requires api_name")
        return self


class GradioToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    ui_label: Optional[str] = None
    enabled: bool = True
    gradio_url: Optional[HttpUrl] = None
    auth: AuthConfig = Field(default_factory=AuthConfig)
    catalog_aliases: List[str] = Field(default_factory=list)
    default_endpoint: Optional[str] = None
    timeout_seconds: float = 300.0
    max_download_bytes: int = 1024 * 1024 * 1024
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    endpoints: List[GradioEndpointConfig]

    @field_validator("id", "display_name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field cannot be empty")
        return value.strip()

    @field_validator("catalog_aliases")
    @classmethod
    def _aliases(cls, values: List[str]) -> List[str]:
        cleaned = [_normalize_alias(v) for v in values]
        if any(not v for v in cleaned):
            raise ValueError("catalog_aliases cannot contain empty values")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("catalog_aliases contains duplicate values")
        return cleaned

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        return value

    @field_validator("max_download_bytes")
    @classmethod
    def _download_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_download_bytes must be greater than zero")
        return value

    @model_validator(mode="after")
    def _validate_tool(self) -> "GradioToolConfig":
        if self.enabled and self.gradio_url is None:
            raise ValueError(f"enabled tool {self.id!r} requires gradio_url")
        if not self.endpoints:
            raise ValueError(f"tool {self.id!r} must contain at least one endpoint")
        endpoint_ids = [e.id for e in self.endpoints]
        duplicate_endpoint_ids = sorted({x for x in endpoint_ids if endpoint_ids.count(x) > 1})
        if duplicate_endpoint_ids:
            raise ValueError(f"tool {self.id!r} has duplicate endpoint ids: {duplicate_endpoint_ids}")
        if self.default_endpoint:
            endpoint = next((e for e in self.endpoints if e.id == self.default_endpoint), None)
            if endpoint is None:
                raise ValueError(f"tool {self.id!r} default_endpoint {self.default_endpoint!r} does not exist")
            if not endpoint.enabled:
                raise ValueError(f"tool {self.id!r} default_endpoint {self.default_endpoint!r} is disabled")
        enabled_count = len([e for e in self.endpoints if e.enabled])
        if self.catalog_aliases and self.default_endpoint is None and enabled_count != 1:
            raise ValueError(f"tool {self.id!r} has tool-level catalog_aliases but no unambiguous default_endpoint")
        return self


class GradioToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    tools: List[GradioToolConfig]

    @model_validator(mode="after")
    def _validate_registry_collisions(self) -> "GradioToolsConfig":
        seen_tools: set[str] = set()
        seen_aliases: Dict[str, tuple[str, str, str]] = {}
        for tool in self.tools:
            if tool.id in seen_tools:
                raise ValueError(f"duplicate tool id {tool.id!r}")
            seen_tools.add(tool.id)
            endpoint_ids = {endpoint.id for endpoint in tool.endpoints}
            default_endpoint = tool.default_endpoint or (tool.endpoints[0].id if len(tool.endpoints) == 1 else None)
            for alias in tool.catalog_aliases:
                if not default_endpoint or default_endpoint not in endpoint_ids:
                    raise ValueError(f"tool-level alias {alias!r} on tool {tool.id!r} is ambiguous because no valid default_endpoint exists")
                _record_alias(seen_aliases, alias, tool.id, default_endpoint, "tool.catalog_aliases")
            for endpoint in tool.endpoints:
                for alias in endpoint.catalog_aliases:
                    _record_alias(seen_aliases, alias, tool.id, endpoint.id, "endpoint.catalog_aliases")
        return self


class GenericGradioInput(BaseModel):
    tool_id: str
    endpoint_id: str
    image_path: Optional[str] = None
    image_paths: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class GenericGradioOutput(BaseToolOutput):
    stdout: str = ""


@dataclass
class ToolConfig:
    name: str
    display_name: str
    icon: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    executor: Callable
    catalog_names: Optional[List[str]] = None
    supports_images: bool = True
    supports_files: bool = True
    requires_approval: bool = True
    preview_field: str = "result_preview"
    download_fields: List[str] | str = "result_origin"
    metadata_field: Optional[str] = "metadata_text"
    notes_field: str = "notes"
    success_field: str = "success"
    error_field: str = "error"
    compute_time_field: str = "compute_time_seconds"
    gradio: Optional[GradioToolConfig] = None
    endpoint: Optional[GradioEndpointConfig] = None

    @property
    def endpoint_id(self) -> Optional[str]:
        return self.endpoint.id if self.endpoint else None

    @property
    def gradio_url(self) -> Optional[str]:
        return str(self.gradio.gradio_url) if self.gradio and self.gradio.gradio_url else None

    @property
    def api_name(self) -> Optional[str]:
        return self.endpoint.api_name if self.endpoint else None

    def is_runnable(self) -> bool:
        return bool(self.gradio and self.endpoint and self.gradio.enabled and self.endpoint.enabled and self.gradio.gradio_url and self.endpoint.api_name and self.endpoint.demo.available)


TOOL_REGISTRY: Dict[str, ToolConfig] = {}
CATALOG_NAME_TO_TOOL: Dict[str, str] = {}
CATALOG_NAME_TO_ENDPOINT: Dict[str, tuple[str, str]] = {}
ACTIVE_CONFIG: Optional[GradioToolsConfig] = None
ACTIVE_CONFIG_PATH: Optional[Path] = None
_LOCK = threading.RLock()


def default_config_path() -> Path:
    with resources.as_file(resources.files(DEFAULT_CONFIG_PACKAGE).joinpath(DEFAULT_CONFIG_NAME)) as p:
        return Path(p)


def resolve_config_path(path: Optional[str | Path] = None) -> Path:
    value = path or os.getenv(GRADIO_TOOLS_CONFIG_ENV)
    return Path(value).expanduser().resolve() if value else default_config_path()


def load_config(path: Optional[str | Path] = None) -> GradioToolsConfig:
    config_path = resolve_config_path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(f"Malformed JSON in {config_path}: {exc}") from exc
    except OSError as exc:
        raise RegistryValidationError(f"Could not read Gradio tools config {config_path}: {exc}") from exc
    return validate_config_payload(raw)


def validate_config_payload(payload: Dict[str, Any]) -> GradioToolsConfig:
    try:
        return GradioToolsConfig.model_validate(payload)
    except ValidationError as exc:
        raise RegistryValidationError(_format_pydantic_errors(exc)) from exc
    except ValueError as exc:
        raise RegistryValidationError(str(exc)) from exc


def save_config_payload(payload: Dict[str, Any], path: Optional[str | Path] = None) -> Path:
    config = validate_config_payload(payload)
    config_path = resolve_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(config.model_dump(mode="json", exclude_none=True), indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=config_path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, config_path)
    return config_path


def initialize_registry(path: Optional[str | Path] = None, *, force: bool = False) -> None:
    config_path = resolve_config_path(path)
    with _LOCK:
        if not force and ACTIVE_CONFIG is not None and ACTIVE_CONFIG_PATH == config_path:
            return
        config = load_config(config_path)
        registries = _build_registries(config)
        _swap_registry(config, config_path, registries)


def reload_registry(path: Optional[str | Path] = None) -> None:
    initialize_registry(path, force=True)


def active_config_json() -> Dict[str, Any]:
    initialize_registry()
    with _LOCK:
        assert ACTIVE_CONFIG is not None
        return ACTIVE_CONFIG.model_dump(mode="json", exclude_none=True)


def active_config_path() -> Path:
    initialize_registry()
    with _LOCK:
        assert ACTIVE_CONFIG_PATH is not None
        return ACTIVE_CONFIG_PATH


def register_tool(config: ToolConfig) -> None:
    with _LOCK:
        if config.name in TOOL_REGISTRY:
            raise ValueError(f"Tool {config.name!r} is already registered")
        TOOL_REGISTRY[config.name] = config
        for catalog_name in config.catalog_names or []:
            alias = _normalize_alias(catalog_name)
            if alias in CATALOG_NAME_TO_TOOL and CATALOG_NAME_TO_TOOL[alias] != config.name:
                raise ValueError(f"Catalog alias {alias!r} already registered")
            CATALOG_NAME_TO_TOOL[alias] = config.name
            if config.endpoint:
                CATALOG_NAME_TO_ENDPOINT[alias] = (config.name, config.endpoint.id)


def get_tool(name: str, endpoint_id: Optional[str] = None) -> Optional[ToolConfig]:
    initialize_registry()
    with _LOCK:
        if endpoint_id:
            return TOOL_REGISTRY.get(_registry_key(name, endpoint_id))
        config = TOOL_REGISTRY.get(name)
        if config:
            return config
        match = CATALOG_NAME_TO_ENDPOINT.get(_normalize_alias(name))
        if match:
            return TOOL_REGISTRY.get(_registry_key(match[0], match[1]))
    return None


def resolve_catalog_alias(alias: str) -> Optional[ToolConfig]:
    return get_tool(alias)


def resolve_runnable_url(url: str) -> Optional[ToolConfig]:
    """Resolve a catalog runnableExample URL to a configured Gradio tool."""
    needle = _normalize_runnable_url(url)
    if not needle:
        return None
    initialize_registry()
    with _LOCK:
        for name, config in TOOL_REGISTRY.items():
            if ":" in name or not config.gradio_url:
                continue
            if _normalize_runnable_url(config.gradio_url) == needle:
                return config
    return None


def alias_is_tool_level(alias: str, tool: ToolConfig) -> bool:
    if not tool.gradio:
        return False
    normalized = _normalize_alias(alias)
    return normalized == _normalize_alias(tool.name) or normalized in {
        _normalize_alias(a) for a in tool.gradio.catalog_aliases
    }


def list_tool_endpoints(tool_name: str) -> List[ToolConfig]:
    initialize_registry()
    with _LOCK:
        tool = TOOL_REGISTRY.get(tool_name)
        if not tool or not tool.gradio:
            return []
        endpoints: List[ToolConfig] = []
        for endpoint in tool.gradio.endpoints:
            config = TOOL_REGISTRY.get(_registry_key(tool.name, endpoint.id))
            if config:
                endpoints.append(config)
        return endpoints


def list_tools() -> List[str]:
    initialize_registry()
    with _LOCK:
        return [name for name in TOOL_REGISTRY if ":" not in name]


def list_configured_tools() -> List[ToolConfig]:
    initialize_registry()
    with _LOCK:
        return [config for name, config in TOOL_REGISTRY.items() if ":" not in name]


def get_tool_display_name(name: str, endpoint_id: Optional[str] = None) -> str:
    tool = get_tool(name, endpoint_id)
    if tool:
        if endpoint_id and tool.endpoint:
            return tool.endpoint.display_name
        return tool.display_name
    return name.replace("_", " ").title()


def get_tool_icon(name: str) -> str:
    tool = get_tool(name)
    return tool.icon if tool else "T"


def extract_output_field(output: BaseModel, field_name: str) -> Any:
    return getattr(output, field_name, None)


def extract_preview(output: BaseModel, tool_name: str) -> Optional[str]:
    tool = get_tool(tool_name)
    return extract_output_field(output, tool.preview_field) if tool else None


def extract_downloads(output: BaseModel, tool_name: str) -> List[str]:
    tool = get_tool(tool_name)
    if not tool:
        return []
    fields = tool.download_fields if isinstance(tool.download_fields, list) else [tool.download_fields]
    downloads: List[str] = []
    for field in fields:
        value = extract_output_field(output, field)
        if isinstance(value, list):
            downloads.extend([v for v in value if v])
        elif isinstance(value, str) and value:
            downloads.append(value)
    return downloads


def extract_metadata(output: BaseModel, tool_name: str) -> Optional[str]:
    tool = get_tool(tool_name)
    if not tool or not tool.metadata_field:
        return None
    return extract_output_field(output, tool.metadata_field)


def execute_configured_gradio_endpoint(inp: GenericGradioInput) -> GenericGradioOutput:
    from ai_agent.agent.tools.mcp.gradio_executor import execute_gradio_endpoint

    return execute_gradio_endpoint(inp)


def _build_registries(config: GradioToolsConfig) -> tuple[Dict[str, ToolConfig], Dict[str, str], Dict[str, tuple[str, str]]]:
    tools: Dict[str, ToolConfig] = {}
    aliases: Dict[str, str] = {}
    endpoints_by_alias: Dict[str, tuple[str, str]] = {}
    for tool in config.tools:
        default_endpoint_id = tool.default_endpoint or (tool.endpoints[0].id if len(tool.endpoints) == 1 else None)
        tools[tool.id] = ToolConfig(
            name=tool.id,
            display_name=tool.display_name,
            icon=tool.icon or tool.ui_label or "T",
            input_model=GenericGradioInput,
            output_model=GenericGradioOutput,
            executor=execute_configured_gradio_endpoint,
            catalog_names=tool.catalog_aliases,
            requires_approval=True,
            gradio=tool,
            endpoint=next((e for e in tool.endpoints if e.id == default_endpoint_id), None),
        )
        for endpoint in tool.endpoints:
            tools[_registry_key(tool.id, endpoint.id)] = ToolConfig(
                name=tool.id,
                display_name=tool.display_name,
                icon=tool.icon or tool.ui_label or "T",
                input_model=GenericGradioInput,
                output_model=GenericGradioOutput,
                executor=execute_configured_gradio_endpoint,
                catalog_names=list(dict.fromkeys(tool.catalog_aliases + endpoint.catalog_aliases)),
                requires_approval=endpoint.approval.required,
                gradio=tool,
                endpoint=endpoint,
            )
        for alias in tool.catalog_aliases:
            if default_endpoint_id is None:
                raise RegistryValidationError(f"tool {tool.id!r} alias {alias!r} is ambiguous without default_endpoint")
            aliases[alias] = tool.id
            endpoints_by_alias[alias] = (tool.id, default_endpoint_id)
        for endpoint in tool.endpoints:
            for alias in endpoint.catalog_aliases:
                aliases[alias] = tool.id
                endpoints_by_alias[alias] = (tool.id, endpoint.id)
    return tools, aliases, endpoints_by_alias


def _swap_registry(config: GradioToolsConfig, path: Path, registries: tuple[Dict[str, ToolConfig], Dict[str, str], Dict[str, tuple[str, str]]]) -> None:
    global ACTIVE_CONFIG, ACTIVE_CONFIG_PATH
    tools, aliases, endpoint_aliases = registries
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(tools)
    CATALOG_NAME_TO_TOOL.clear()
    CATALOG_NAME_TO_TOOL.update(aliases)
    CATALOG_NAME_TO_ENDPOINT.clear()
    CATALOG_NAME_TO_ENDPOINT.update(endpoint_aliases)
    ACTIVE_CONFIG = config
    ACTIVE_CONFIG_PATH = path


def _record_alias(seen: Dict[str, tuple[str, str, str]], alias: str, tool_id: str, endpoint_id: str, field: str) -> None:
    normalized = _normalize_alias(alias)
    existing = seen.get(normalized)
    current = (tool_id, endpoint_id, field)
    if existing and existing[:2] != current[:2]:
        raise ValueError(
            f"duplicate alias {normalized!r}: {existing[2]} maps to tool {existing[0]!r} endpoint {existing[1]!r}, "
            f"but {field} maps to tool {tool_id!r} endpoint {endpoint_id!r}"
        )
    seen[normalized] = current


def _normalize_alias(value: str) -> str:
    return (value or "").strip().casefold()


def _normalize_runnable_url(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.rstrip("/").casefold()
    scheme = (parsed.scheme or "https").casefold()
    host = (parsed.netloc or "").casefold()
    path_parts = [p for p in parsed.path.split("/") if p]

    if host == "huggingface.co":
        if len(path_parts) >= 3 and path_parts[0] == "spaces":
            host = f"{path_parts[1]}-{path_parts[2]}".casefold() + ".hf.space"
            return f"https://{host}"
        if len(path_parts) >= 2:
            host = f"{path_parts[0]}-{path_parts[1]}".casefold() + ".hf.space"
            return f"https://{host}"

    if host.endswith(".hf.space"):
        return f"https://{host}"

    path = (parsed.path or "").rstrip("/")
    return urlunparse((scheme, host, path, "", "", "")).casefold()


def _registry_key(tool_id: str, endpoint_id: str) -> str:
    return f"{tool_id}:{endpoint_id}"


def _format_pydantic_errors(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


def _looks_like_secret(value: str) -> bool:
    if "=" in value or value.startswith(("sk-", "hf_", "ghp_")):
        return True
    return bool(_SECRET_FIELD_RE.search(value) and len(value) > 40)
