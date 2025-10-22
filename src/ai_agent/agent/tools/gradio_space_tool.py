from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
from .utils import get_pipeline
from utils.utils import _best_runnable_link
from gradio_client import Client, handle_file

# -------- Gradio run_example tool -------------------------------------------
class RunExampleInput(BaseModel):
    tool_name: str
    image_path: Optional[str] = None  # absolute or workspace path
    endpoint_url: Optional[str] = None  # direct Space URL override
    extra_text: Optional[str] = None   # optional prompt/caption

class RunExampleOutput(BaseModel):
    tool_name: str
    ran: bool = False
    stdout: str = ""
    endpoint_url: Optional[str] = None
    api_name: Optional[str] = None
    notes: Optional[str] = None


def _choose_endpoint(endpoints: List[Dict[str, Any]], have_image: bool) -> Optional[Dict[str, Any]]:
    """Pick a sensible endpoint: prefer one that accepts an image if we have one; else the first text-only."""
    def has_image(f: Dict[str, Any]) -> bool:
        for i in f.get("inputs", []):
            t = str(i.get("type") or i.get("component") or "").lower()
            if "image" in t:
                return True
        return False

    if have_image:
        for f in endpoints:
            if has_image(f):
                return f
    # fallback: any endpoint
    return endpoints[0] if endpoints else None


def _build_payload(fn: Dict[str, Any], image_path: Optional[str], extra_text: Optional[str]) -> List[Any]:
    inputs = fn.get("inputs", [])
    payload: List[Any] = []
    for spec in inputs:
        t = str(spec.get("type") or spec.get("component") or "").lower()
        # Gradio client supports passing file paths for image inputs
        if "image" in t and image_path:
            payload.append(handle_file(image_path) if handle_file else image_path)
        elif "textbox" in t or "text" in t or "textarea" in t:
            payload.append(extra_text or "")
        else:
            # default empty for other inputs (checkbox, number, etc.)
            payload.append("")
    return payload


def tool_run_example(inp: RunExampleInput) -> RunExampleOutput:
    """Run a remote Gradio demo for a catalog tool on an optional user image using gradio_client.

    Behavior:
      - Determine Space URL: prefer explicit endpoint_url, else catalog runnable link.
      - Discover API endpoints via view_api and choose one matching image/no-image needs.
      - Build payload by mapping image path to image inputs and extra_text into text fields.
    """
    pipe = get_pipeline()
    url = (inp.endpoint_url or None)
    if not url:
        doc = pipe.get_doc(inp.tool_name)
        if doc:
            url = _best_runnable_link(doc)
    if not url:
        return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes="No runnable example URL found")

    try:
        client = Client(url)
        apis = client.view_api(return_format="dict") or {}
        endpoints = apis.get("endpoints") or apis.get("named_endpoints") or []
        if not isinstance(endpoints, list):
            # some versions return dict of name->spec
            endpoints = list(endpoints.values())
        fn = _choose_endpoint(endpoints, have_image=bool(inp.image_path))
        if not fn:
            return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes="No endpoints discovered", endpoint_url=url)
        api_name = fn.get("api_name") or fn.get("path") or fn.get("route")
        if not api_name:
            # common default
            api_name = "/predict"
        payload = _build_payload(fn, inp.image_path, inp.extra_text)
        try:
            res = client.predict(*payload, api_name=api_name)
            stdout = str(res)
        except Exception as e:
            return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes=f"predict failed: {e}", endpoint_url=url, api_name=api_name)
        return RunExampleOutput(tool_name=inp.tool_name, ran=True, stdout=str(stdout)[:6000], endpoint_url=url, api_name=str(api_name))
    except Exception as e:
        return RunExampleOutput(tool_name=inp.tool_name, ran=False, notes=str(e), endpoint_url=url)