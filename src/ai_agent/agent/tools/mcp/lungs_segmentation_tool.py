from __future__ import annotations

from typing import Optional, Any, Dict, Tuple
import os
import logging
import tempfile
from pathlib import Path
import time

import requests
from gradio_client import Client, handle_file

from ai_agent.utils.previews import _build_preview_for_vlm
from ai_agent.utils.temp_file_manager import register_temp_file
from ai_agent.agent.tools.mcp.registry import register_tool, ToolConfig
from ai_agent.agent.tools.mcp.base import BaseToolOutput, ImageToolInput

log = logging.getLogger("agent.lungs_segmentation")


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class LungsSegmentationInput(ImageToolInput):
    """Input for 3D lungs segmentation tool."""

    pass  # Inherits image_path and description from ImageToolInput


class LungsSegmentationOutput(BaseToolOutput):
    """Output from 3D lungs segmentation tool."""

    # All standard fields inherited from BaseToolOutput:
    # - success, error, compute_time_seconds, notes
    # - result_preview, result_origin, result_path
    # - metadata_text, endpoint_url, api_name
    pass


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
LUNGS_SEGMENTATION_ENDPOINT = "https://qchapp-3d-lungs-segmentation.hf.space/"
LUNGS_SEGMENTATION_API_NAME = "/segment"

# Maximum file size for downloads (1GB for medical imaging)
MAX_DOWNLOAD_SIZE = 1024 * 1024 * 1024  # 1GB in bytes


# ---------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------
def tool_lungs_segmentation(inp: LungsSegmentationInput) -> LungsSegmentationOutput:
    """
    Run 3D lungs segmentation on a CT scan image via a Gradio Space.

    Materialization strategy (robust):
      1) If Space returns dict FileData (url/path/etc) -> download via URL.
      2) If Space returns URL string -> download.
      3) If Space returns local file -> use it.
      4) If Space returns server path (/tmp/...) -> try /gradio_api/file=... (may 403).
    """
    start_time = time.time()

    if not os.path.exists(inp.image_path):
        return LungsSegmentationOutput(
            success=False,
            error=f"Image file not found: {inp.image_path}",
            endpoint_url=LUNGS_SEGMENTATION_ENDPOINT,
            api_name=LUNGS_SEGMENTATION_API_NAME,
        )

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    try:
        log.info(
            "Running lungs segmentation on %s (endpoint: %s)",
            inp.image_path,
            LUNGS_SEGMENTATION_ENDPOINT,
        )

        client = _make_gradio_client(LUNGS_SEGMENTATION_ENDPOINT, hf_token)

        # Call API
        try:
            result = client.predict(
                file_obj=handle_file(inp.image_path),
                api_name=LUNGS_SEGMENTATION_API_NAME,
            )
            log.info("API returned type=%s value=%r", type(result), result)
        except Exception as e:
            return LungsSegmentationOutput(
                success=False,
                error=f"API call failed: {e}",
                compute_time_seconds=time.time() - start_time,
                endpoint_url=LUNGS_SEGMENTATION_ENDPOINT,
                api_name=LUNGS_SEGMENTATION_API_NAME,
            )

        # Materialize to local file
        origin_path = _materialize_any(result, client=client, hf_token=hf_token)

        compute_time = time.time() - start_time

        if not origin_path or not os.path.exists(origin_path):
            # This is the common case if the Space returns '/tmp/...' and Gradio blocks it (403).
            return LungsSegmentationOutput(
                success=False,
                error="Could not materialize/download the result file.",
                compute_time_seconds=compute_time,
                endpoint_url=LUNGS_SEGMENTATION_ENDPOINT,
                api_name=LUNGS_SEGMENTATION_API_NAME,
                notes=(
                    f"API returned: {result!r}. If this is a '/tmp/...' path and you see HTTP 403, "
                    "the Space must return a FileData/url (recommended) or whitelist the output directory "
                    "via allowed_paths / GRADIO_TEMP_DIR."
                ),
            )

        # Build preview + metadata using your shared function
        preview_path, meta_text = _safe_build_preview(origin_path)

        # Back-compat: prefer preview in result_path
        result_path = preview_path or origin_path

        return LungsSegmentationOutput(
            success=True,
            result_path=result_path,
            result_origin=origin_path,
            result_preview=preview_path,
            metadata_text=meta_text,
            compute_time_seconds=compute_time,
            endpoint_url=LUNGS_SEGMENTATION_ENDPOINT,
            api_name=LUNGS_SEGMENTATION_API_NAME,
            notes=f"Successfully segmented lungs from {os.path.basename(inp.image_path)}",
        )

    except Exception as e:
        log.exception("Lungs segmentation failed")
        return LungsSegmentationOutput(
            success=False,
            error=str(e),
            compute_time_seconds=time.time() - start_time,
            endpoint_url=LUNGS_SEGMENTATION_ENDPOINT,
            api_name=LUNGS_SEGMENTATION_API_NAME,
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _make_gradio_client(endpoint: str, hf_token: Optional[str]) -> Client:
    """
    Create a gradio_client.Client with best compatibility across versions.
    """
    # Set extended timeout for both connection and operations (5 minutes for large files)
    httpx_kwargs = {"timeout": 300.0}

    # Newer versions use token=, older versions used hf_token=
    if hf_token:
        try:
            return Client(endpoint, hf_token=hf_token, httpx_kwargs=httpx_kwargs)
        except TypeError:
            # Fallback for very old versions without httpx_kwargs support
            try:
                return Client(endpoint, hf_token=hf_token)
            except TypeError:
                return Client(endpoint)

    try:
        return Client(endpoint, httpx_kwargs=httpx_kwargs)
    except TypeError:
        # Fallback for very old versions
        return Client(endpoint)


def _safe_build_preview(origin_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Wrapper around _build_preview_for_vlm so preview failures never break the tool.
    """
    try:
        preview_path, meta_text = _build_preview_for_vlm([origin_path])
        return preview_path, meta_text
    except Exception as e:
        log.debug("Preview build failed for %s: %r", origin_path, e)
        return None, None


def _materialize_any(
    obj: Any, client: Client, hf_token: Optional[str] = None, _depth: int = 0
) -> Optional[str]:
    """
    Convert common Gradio outputs into a local file path.

    Supported:
      - local path string
      - URL string
      - dict (FileData-like) containing url/path/name/filepath
      - list/tuple containing any of the above
      - server path '/tmp/...' -> attempt Gradio file endpoint (may 403)

    Args:
        obj: Object to materialize
        client: Gradio client
        hf_token: Optional HuggingFace token
        _depth: Internal recursion depth counter (max 10)
    """
    if obj is None or _depth > 10:
        if _depth > 10:
            log.warning("Recursion depth limit reached in _materialize_any, halting.")
        return None

    # list/tuple: most Gradio outputs are single-element lists
    if isinstance(obj, (list, tuple)) and obj:
        return _materialize_any(
            obj[0], client=client, hf_token=hf_token, _depth=_depth + 1
        )

    # dict: FileData-like is best case (url provided)
    if isinstance(obj, dict):
        # Prefer URL if present
        url = obj.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            log.info("Materialize: dict url=%s", url)
            return _download_to_temp(url, hf_token=hf_token)

        # Fall back through common keys
        for k in ("path", "filepath", "file", "name"):
            v = obj.get(k)
            if isinstance(v, str) and v:
                return _materialize_any(
                    v, client=client, hf_token=hf_token, _depth=_depth + 1
                )

        return None

    # string: local file, URL, or server path
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return None

        # local file?
        p = Path(s)
        if p.exists() and p.is_file():
            log.info("Materialize: local file=%s", s)
            return str(p)

        # URL?
        if s.startswith(("http://", "https://")):
            log.info("Materialize: url=%s", s)
            return _download_to_temp(s, hf_token=hf_token)

        # server path? (e.g. /tmp/xxx_mask.tif)
        if s.startswith("/"):
            log.info("Materialize: server path=%s", s)
            return _download_from_gradio_file_endpoint(client, s, hf_token=hf_token)

    return None


def _download_to_temp(url: str, hf_token: Optional[str] = None) -> Optional[str]:
    """
    Download a URL to a temporary file (streaming) with size limit checks.
    """
    headers: Dict[str, str] = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    try:
        with requests.get(
            url, headers=headers, timeout=120, stream=True, allow_redirects=True
        ) as r:
            if r.status_code != 200:
                log.error("Download failed: url=%s status=%s", url, r.status_code)
                return None

            # Check Content-Length if available
            content_length = r.headers.get("content-length")
            if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
                log.error(
                    "File too large: %s bytes (max %s)",
                    content_length,
                    MAX_DOWNLOAD_SIZE,
                )
                return None

            ext = _guess_ext(url, r.headers.get("content-type", ""))

            with tempfile.NamedTemporaryFile(
                delete=False, prefix="lungs_seg_", suffix=ext
            ) as f:
                downloaded_size = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > MAX_DOWNLOAD_SIZE:
                            log.error(
                                "Download exceeded size limit: %s bytes",
                                downloaded_size,
                            )
                            f.close()
                            os.remove(f.name)
                            return None
                        f.write(chunk)
                log.info("Downloaded %s bytes: %s -> %s", downloaded_size, url, f.name)
                return register_temp_file(f.name)
    except Exception as e:
        log.error("Failed to download %s: %r", url, e)
        return None


def _download_from_gradio_file_endpoint(
    client: Client, server_path: str, hf_token: Optional[str] = None
) -> Optional[str]:
    """
    Last-resort fallback when API returns '/tmp/...' but no URL.
    Often blocked with 403 unless Space allows that directory or writes into Gradio temp/cache.
    Includes size limit checks.
    """
    base = (getattr(client, "src", None) or LUNGS_SEGMENTATION_ENDPOINT).rstrip("/")
    file_url = f"{base}/gradio_api/file={server_path}"

    headers: Dict[str, str] = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    params: Dict[str, str] = {}
    session_hash = getattr(client, "session_hash", None)
    if session_hash:
        params["session_hash"] = session_hash

    try:
        r = requests.get(
            file_url, headers=headers, params=params, timeout=60, stream=True
        )
        if r.status_code == 403:
            # Common: file exists but not allowed to be served
            detail: Any
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:200]
            log.error("HTTP 403 from %s detail=%r", file_url, detail)
            return None

        if r.status_code != 200:
            log.error("HTTP %s from %s", r.status_code, file_url)
            return None

        # Check Content-Length before downloading
        content_length = r.headers.get("content-length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            log.error(
                "File too large: %s bytes (max %s)", content_length, MAX_DOWNLOAD_SIZE
            )
            return None

        # Read content with size check
        content = b""
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                content += chunk
                if len(content) > MAX_DOWNLOAD_SIZE:
                    log.error("Download exceeded size limit: %s bytes", len(content))
                    return None

        ct = r.headers.get("content-type", "")
        if "html" in ct.lower() or content.startswith(b"<!"):
            log.error("Got HTML instead of file from %s", file_url)
            return None

        ext = os.path.splitext(server_path)[1] or ".tif"
        with tempfile.NamedTemporaryFile(
            delete=False, prefix="lungs_seg_", suffix=ext
        ) as f:
            f.write(content)
            log.info(
                "Downloaded %s bytes from gradio file endpoint -> %s",
                len(content),
                f.name,
            )
            return register_temp_file(f.name)

    except Exception as e:
        log.error("Failed gradio file endpoint download: %r", e)
        return None


def _guess_ext(url: str, content_type: str) -> str:
    """
    Guess file extension from URL path or Content-Type.
    """
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


# ---------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------
register_tool(
    ToolConfig(
        name="lungs_segmentation",
        display_name="3D Lungs Segmentation",
        icon="🫁",
        catalog_names=["lungs-segmentation"],  # Catalog name from dataset/catalog.jsonl
        input_model=LungsSegmentationInput,
        output_model=LungsSegmentationOutput,
        executor=tool_lungs_segmentation,
        supports_images=True,
        supports_files=True,
        requires_approval=True,
        preview_field="result_preview",
        download_fields="result_origin",  # Could also be ["result_origin", "other_file"]
        metadata_field="metadata_text",
        notes_field="notes",
        success_field="success",
        error_field="error",
        compute_time_field="compute_time_seconds",
    )
)
