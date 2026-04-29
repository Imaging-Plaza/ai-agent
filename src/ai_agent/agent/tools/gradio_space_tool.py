from __future__ import annotations

from typing import Optional, Any
from pydantic import BaseModel
import os
import re
import logging
from .utils import get_pipeline
from ai_agent.utils.utils import _best_runnable_link
from ai_agent.utils.previews import _build_preview_for_vlm
from ai_agent.utils.temp_file_manager import register_temp_file
from gradio_client import Client, handle_file
import tempfile
from pathlib import Path
import requests


# -------- Gradio run_example tool -------------------------------------------
class RunExampleInput(BaseModel):
    tool_name: str
    image_path: Optional[str] = None  # absolute or workspace path
    endpoint_url: Optional[str] = None  # direct Space URL override
    extra_text: Optional[str] = None  # optional prompt/caption


class RunExampleOutput(BaseModel):
    tool_name: str
    ran: bool = False
    stdout: str = ""
    endpoint_url: Optional[str] = None
    api_name: Optional[str] = None
    notes: Optional[str] = None
    # Back-compat: 'result_image' kept as alias for preview
    result_image: Optional[str] = None
    result_preview: Optional[str] = None
    result_origin: Optional[str] = None  # original returned file (downloaded if URL)


log = logging.getLogger("agent.run_example")

_HF_SPACE_RE = re.compile(r"^https?://huggingface\.co/spaces/([^/]+)/([^/]+)/?$")


def _normalize_space_identifier(url_or_name: str) -> str:
    """Accepts full HF Spaces URL or 'owner/space' or a direct app URL; returns a Client-acceptable src.
    Prefer 'owner/space' for HF Spaces page URLs.
    """
    s = (url_or_name or "").strip()
    m = _HF_SPACE_RE.match(s)
    if m:
        owner, space = m.group(1), m.group(2)
        return f"{owner}/{space}"
    return s


def _download_to_temp(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200 or not r.content:
            return None
        # try to preserve extension from URL
        from urllib.parse import urlparse

        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1]
        if not ext:
            # guess based on content-type
            ct = r.headers.get("content-type", "").lower()
            if "tiff" in ct or "tif" in ct:
                ext = ".tif"
            elif "png" in ct:
                ext = ".png"
            elif "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            else:
                ext = ".bin"
        with tempfile.NamedTemporaryFile(
            delete=False, prefix="demo_result_", suffix=ext
        ) as fd:
            fd.write(r.content)
            fd.flush()
            return register_temp_file(fd.name)
    except Exception:
        return None


def _materialize_result(obj: Any) -> Optional[str]:
    """Try to materialize an image result to a local file and return the path.
    Accepts a filepath or URL from common Gradio outputs.
    """
    # Direct file path
    try:
        s = str(obj)
    except Exception:
        return None
    if not s:
        return None
    # If it's an existing local file
    p = Path(s)
    if p.exists() and p.is_file():
        return str(p)
    # If it's a URL, try to download
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        return _download_to_temp(s)
    # Unknown shape
    return None


def tool_run_example(inp: RunExampleInput) -> RunExampleOutput:
    """Run a remote Gradio demo for a catalog tool on an optional user image using gradio_client.

    Behavior:
      - Determine Space URL: prefer explicit endpoint_url, else catalog runnable link.
      - Used agreed endpoint /segment for now
      - Build payload by mapping image path to image inputs and extra_text into text fields.
    """
    pipe = get_pipeline()
    url = inp.endpoint_url or None
    if not url:
        doc = pipe.get_doc(inp.tool_name)
        if doc:
            url = _best_runnable_link(doc)
    if not url:
        return RunExampleOutput(
            tool_name=inp.tool_name, ran=False, notes="No runnable example URL found"
        )

    try:
        src = _normalize_space_identifier(url)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        log.info(
            "Gradio run_example: src=%s (from=%s), tool=%s", src, url, inp.tool_name
        )
        client = Client(src, hf_token=hf_token) if hf_token else Client(src)
        api_name = "/segment"  # agreed endpoint
        # For a simple segmentation endpoint that takes a single image input
        if inp.image_path:
            payload_file = handle_file(inp.image_path)
            payload = [payload_file]
            try:
                log.info(
                    "Gradio run_example payload: file=%s ext=%s",
                    inp.image_path,
                    os.path.splitext(inp.image_path)[1].lower(),
                )
            except Exception:
                pass
        else:
            payload = [""]
        try:
            # Prefer keyword expected by docs ('file_obj'), then fallback to positional
            try:
                res = client.predict(file_obj=payload[0], api_name=api_name)
            except Exception as e_kw:
                log.debug(
                    "Keyword predict failed, falling back to positional: %r", e_kw
                )
                res = client.predict(*payload, api_name=api_name)
            stdout = str(res)
        except Exception as e:
            return RunExampleOutput(
                tool_name=inp.tool_name,
                ran=False,
                notes=f"predict failed: {e}",
                endpoint_url=url,
                api_name=api_name,
            )

        # Materialize original result file (any supported format)
        origin_path = None
        if isinstance(res, (list, tuple)) and res:
            origin_path = _materialize_result(res[0]) or origin_path
        elif isinstance(res, dict):
            # common keys from outputs
            for k in ("file", "filepath", "image", "output", "result", "mask"):
                if k in res:
                    origin_path = _materialize_result(res[k]) or origin_path
                    if origin_path:
                        break
        else:
            origin_path = _materialize_result(res)

        preview_path = None
        if origin_path:
            try:
                preview_path, _ = _build_preview_for_vlm([origin_path])
            except Exception as e:
                log.debug("Preview build failed for %s: %r", origin_path, e)

        return RunExampleOutput(
            tool_name=inp.tool_name,
            ran=True,
            stdout=str(stdout)[:6000],
            endpoint_url=url,
            api_name=str(api_name),
            result_image=preview_path,
            result_preview=preview_path,
            result_origin=origin_path,
        )
    except Exception as e:
        return RunExampleOutput(
            tool_name=inp.tool_name, ran=False, notes=str(e), endpoint_url=url
        )
