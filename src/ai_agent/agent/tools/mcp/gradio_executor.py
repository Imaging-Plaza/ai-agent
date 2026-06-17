from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from gradio_client import Client, handle_file

from ai_agent.agent.tools.mcp.registry import GenericGradioInput, GenericGradioOutput, OutputSelector, get_tool
from ai_agent.utils.previews import _build_preview_for_vlm
from ai_agent.utils.temp_file_manager import register_temp_file

log = logging.getLogger("agent.gradio_executor")


def execute_gradio_endpoint(inp: GenericGradioInput) -> GenericGradioOutput:
    started = time.time()
    tool = get_tool(inp.tool_id, inp.endpoint_id)
    if not tool or not tool.gradio or not tool.endpoint:
        return _failure(inp, "unknown_tool_or_endpoint", started)
    if not tool.gradio.enabled:
        return _failure(inp, f"Tool {inp.tool_id!r} is disabled", started, tool=tool)
    if not tool.endpoint.enabled:
        return _failure(inp, f"Endpoint {inp.endpoint_id!r} is disabled", started, tool=tool)
    if not tool.gradio.gradio_url:
        return _failure(inp, f"Tool {inp.tool_id!r} has no Gradio URL", started, tool=tool)
    if not tool.endpoint.api_name:
        return _failure(inp, f"Endpoint {inp.endpoint_id!r} has no Gradio api_name", started, tool=tool)

    hf_token = _resolve_token(tool.gradio.auth)
    if tool.gradio.auth.required and not hf_token:
        envs = ", ".join(tool.gradio.auth.candidate_envs()) or "<none configured>"
        return _failure(inp, f"Missing authentication environment variable ({envs})", started, tool=tool)

    try:
        args, kwargs = _build_inputs(tool, inp)
    except ValueError as exc:
        return _failure(inp, str(exc), started, tool=tool)

    try:
        client = _make_client(str(tool.gradio.gradio_url), hf_token, tool.endpoint.timeout_seconds or tool.gradio.timeout_seconds)
    except Exception as exc:
        return _failure(inp, f"Gradio connection failed: {exc}", started, tool=tool)

    try:
        if tool.endpoint.input_mapping.call_style == "positional":
            response = client.predict(*args, api_name=tool.endpoint.api_name)
        else:
            response = client.predict(**kwargs, api_name=tool.endpoint.api_name)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "auth" in msg.lower():
            msg = f"Authentication failed or access denied: {msg}"
        return _failure(inp, f"Gradio endpoint call failed: {msg}", started, tool=tool)

    output_mapping = tool.endpoint.output_mapping
    max_bytes = tool.endpoint.max_download_bytes or tool.gradio.max_download_bytes
    origin = _select_and_materialize(response, output_mapping.original, client, hf_token, max_bytes)
    preview = _select_and_materialize(response, output_mapping.preview, client, hf_token, max_bytes)
    metadata_text = _extract_text(response, output_mapping.metadata)
    notes = _extract_text(response, output_mapping.notes)

    if output_mapping.preview.build_preview and not preview and origin:
        try:
            preview, generated_meta = _build_preview_for_vlm([origin])
            metadata_text = metadata_text or generated_meta
        except Exception as exc:
            log.debug("Preview generation failed for %s: %r", origin, exc)
    elif preview and output_mapping.preview.build_preview:
        try:
            built_preview, generated_meta = _build_preview_for_vlm([preview])
            preview = built_preview or preview
            metadata_text = metadata_text or generated_meta
        except Exception as exc:
            log.debug("Preview generation failed for %s: %r", preview, exc)

    success = _extract_bool(response, output_mapping.success)
    if success is None:
        success = bool(origin or preview) or response is not None
    error = _extract_text(response, output_mapping.error)
    compute_time = _extract_float(response, output_mapping.compute_time)
    elapsed = time.time() - started

    if success and not (origin or preview):
        return _failure(inp, "Missing configured output from Gradio response", started, tool=tool, stdout=str(response)[:6000])

    return GenericGradioOutput(
        success=bool(success),
        error=None if success else error or "Gradio endpoint returned an unsuccessful response",
        compute_time_seconds=compute_time if compute_time is not None else elapsed,
        result_preview=preview,
        result_origin=origin,
        result_path=preview or origin,
        metadata_text=metadata_text,
        notes=notes or (f"Executed {tool.endpoint.display_name}" if success else None),
        endpoint_url=str(tool.gradio.gradio_url),
        api_name=tool.endpoint.api_name or "",
        stdout=str(response)[:6000],
    )


def _build_inputs(tool, inp: GenericGradioInput) -> tuple[list[Any], Dict[str, Any]]:
    args: list[Any] = []
    kwargs: Dict[str, Any] = {}
    for param in tool.endpoint.input_mapping.parameters:
        value: Any
        if param.source in ("session_file", "image_path"):
            image_paths = inp.image_paths or ([inp.image_path] if inp.image_path else [])
            value = image_paths[param.file_index] if param.file_index < len(image_paths) else None
            if param.required and not value:
                raise ValueError(f"Missing required input {param.name!r}: no uploaded file is available")
            if value and param.as_gradio_file:
                if not os.path.exists(value):
                    raise ValueError(f"Input file does not exist: {value}")
                value = handle_file(value)
        elif param.source == "description":
            value = inp.description
            if param.required and not value:
                raise ValueError(f"Missing required input {param.name!r}: description is empty")
        elif param.source == "literal":
            value = param.value
        elif param.source == "param":
            key = param.param or param.name
            value = inp.params.get(key)
            if param.required and value in (None, ""):
                raise ValueError(f"Missing required input parameter {key!r}")
        else:
            raise ValueError(f"Unsupported input source {param.source!r}")
        if tool.endpoint.input_mapping.call_style == "positional":
            args.append(value)
        else:
            kwargs[param.name] = value
    return args, kwargs


def _make_client(endpoint: str, hf_token: Optional[str], timeout: float) -> Client:
    httpx_kwargs = {"timeout": timeout}
    if hf_token:
        try:
            return Client(endpoint, hf_token=hf_token, httpx_kwargs=httpx_kwargs)
        except TypeError:
            try:
                return Client(endpoint, hf_token=hf_token)
            except TypeError:
                return Client(endpoint)
    try:
        return Client(endpoint, httpx_kwargs=httpx_kwargs)
    except TypeError:
        return Client(endpoint)


def _resolve_token(auth) -> Optional[str]:
    for env_name in auth.candidate_envs():
        value = os.getenv(env_name)
        if value:
            return value
    return None


def _select_and_materialize(obj: Any, selector: OutputSelector, client: Client, hf_token: Optional[str], max_bytes: int) -> Optional[str]:
    selected = _select(obj, selector.selector)
    if selected is None:
        return None
    if not selector.materialize:
        return str(selected)
    return _materialize_any(selected, client, hf_token, max_bytes)


def _select(obj: Any, selector: Optional[str]) -> Any:
    if selector in (None, "", "root"):
        return obj
    if selector == "first":
        if isinstance(obj, (list, tuple)):
            return obj[0] if obj else None
        return obj
    cur = obj
    path = selector[2:] if selector.startswith("$.") else selector
    for part in path.split("."):
        if cur is None:
            return None
        if part == "first":
            cur = cur[0] if isinstance(cur, (list, tuple)) and cur else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, (list, tuple)) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if idx < len(cur) else None
        else:
            cur = getattr(cur, part, None)
    return cur


def _extract_text(obj: Any, selector: Optional[str]) -> Optional[str]:
    if not selector:
        return None
    value = _select(obj, selector)
    if value in (None, ""):
        return None
    return str(value)


def _extract_bool(obj: Any, selector: Optional[str]) -> Optional[bool]:
    if not selector:
        return None
    value = _select(obj, selector)
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if not normalized:
            return None
        if normalized in {"1", "true", "yes", "y", "on", "success", "succeeded"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "failed", "failure", "error"}:
            return False
    return bool(value)


def _extract_float(obj: Any, selector: Optional[str]) -> Optional[float]:
    if not selector:
        return None
    value = _select(obj, selector)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _materialize_any(obj: Any, client: Client, hf_token: Optional[str], max_bytes: int, depth: int = 0) -> Optional[str]:
    if obj is None or depth > 10:
        return None
    if isinstance(obj, (list, tuple)) and obj:
        return _materialize_any(obj[0], client, hf_token, max_bytes, depth + 1)
    if isinstance(obj, dict):
        url = obj.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return _download_to_temp(url, hf_token, max_bytes)
        for key in ("path", "filepath", "file", "name", "image", "output", "result", "mask"):
            value = obj.get(key)
            if value:
                found = _materialize_any(value, client, hf_token, max_bytes, depth + 1)
                if found:
                    return found
        return None
    if isinstance(obj, str):
        value = obj.strip()
        if not value:
            return None
        p = Path(value)
        if p.exists() and p.is_file():
            return str(p)
        if value.startswith(("http://", "https://")):
            return _download_to_temp(value, hf_token, max_bytes)
        if value.startswith("/"):
            return _download_from_gradio_file_endpoint(client, value, hf_token, max_bytes)
    return None


def _download_to_temp(url: str, hf_token: Optional[str], max_bytes: int) -> Optional[str]:
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    try:
        with requests.get(url, headers=headers, timeout=120, stream=True, allow_redirects=True) as r:
            if r.status_code != 200:
                log.error("Download failed: %s status=%s", url, r.status_code)
                return None
            content_length = r.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                log.error("Download too large: %s bytes", content_length)
                return None
            ext = _guess_ext(url, r.headers.get("content-type", ""))
            with tempfile.NamedTemporaryFile(delete=False, prefix="gradio_tool_", suffix=ext) as f:
                size = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        size += len(chunk)
                        if size > max_bytes:
                            f.close()
                            os.remove(f.name)
                            return None
                        f.write(chunk)
                return register_temp_file(f.name)
    except Exception as exc:
        log.error("Failed to download %s: %r", url, exc)
        return None


def _download_from_gradio_file_endpoint(client: Client, server_path: str, hf_token: Optional[str], max_bytes: int) -> Optional[str]:
    base = (getattr(client, "src", None) or "").rstrip("/")
    if not base:
        return None
    file_url = f"{base}/gradio_api/file={server_path}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    params: Dict[str, str] = {}
    session_hash = getattr(client, "session_hash", None)
    if session_hash:
        params["session_hash"] = session_hash
    try:
        with requests.get(file_url, headers=headers, params=params, timeout=60, stream=True) as r:
            if r.status_code != 200:
                log.error("Gradio file endpoint failed: %s status=%s", file_url, r.status_code)
                return None
            content_length = r.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                return None
            ext = os.path.splitext(server_path)[1] or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, prefix="gradio_tool_", suffix=ext) as f:
                size = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        size += len(chunk)
                        if size > max_bytes:
                            f.close()
                            os.remove(f.name)
                            return None
                        f.write(chunk)
                return register_temp_file(f.name)
    except Exception as exc:
        log.error("Failed gradio file endpoint download: %r", exc)
        return None


def _guess_ext(url: str, content_type: str) -> str:
    from urllib.parse import urlparse

    path = urlparse(url).path.lower()
    if path.endswith(".nii.gz"):
        return ".nii.gz"
    ext = os.path.splitext(path)[1]
    if ext:
        return ext
    ct = (content_type or "").lower()
    if "tiff" in ct or "tif" in ct:
        return ".tif"
    if "png" in ct:
        return ".png"
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "gif" in ct:
        return ".gif"
    if "nifti" in ct or "nii" in ct:
        return ".nii.gz"
    return ".bin"


def _failure(inp: GenericGradioInput, error: str, started: float, tool=None, stdout: str = "") -> GenericGradioOutput:
    return GenericGradioOutput(
        success=False,
        error=error,
        compute_time_seconds=time.time() - started,
        endpoint_url=str(tool.gradio.gradio_url) if tool and tool.gradio and tool.gradio.gradio_url else "",
        api_name=tool.endpoint.api_name if tool and tool.endpoint and tool.endpoint.api_name else "",
        stdout=stdout,
    )
