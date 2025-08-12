from __future__ import annotations
from typing import Any, List, Optional, Tuple
import os, shutil, logging, time, base64, io
from pathlib import Path
from gradio_client import Client, handle_file

log = logging.getLogger("space_runner")

class SpaceRunError(RuntimeError):
    ...

# Accept more formats returned by Spaces (incl. WEBP)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")

def _pick_image_like(outputs: List[Any]) -> Optional[str]:
    """
    Try to locate an image-like thing in the outputs.
    Returns either a local path or a 'data:image/...' string.
    """
    for o in outputs:
        if isinstance(o, str):
            lo = o.lower()
            if lo.startswith("data:image/"):
                return o
            if lo.endswith(IMAGE_EXTS):
                return o
        if isinstance(o, dict):
            for k in ("image", "path", "file", "url", "output"):
                v = o.get(k)
                if isinstance(v, str):
                    lv = v.lower()
                    if lv.startswith("data:image/"):
                        return v
                    if lv.endswith(IMAGE_EXTS):
                        return v
    return None

def _norm_status(st) -> str:
    """
    Normalize gradio_client Job.status() to an uppercase string like 'FINISHED'.
    Handles enums (Status.FINISHED) and older/newer client return types.
    """
    try:
        v = getattr(st, "code", None)
        if v is None:
            v = getattr(st, "name", None) or getattr(st, "value", None)
        if v is None:
            v = str(st)
    except Exception:
        v = str(st)
    return str(v).upper().replace("STATUS.", "").strip()

def _wait_job(job, timeout: int) -> list:
    start, last = time.time(), 0.0
    while True:
        try:
            st = job.status()
            code = _norm_status(st)
        except Exception:
            code = "UNKNOWN"

        if code in {"FINISHED", "SUCCESS", "COMPLETE", "DONE"}:
            # different client versions expose outputs/result
            try:
                out = job.outputs()
            except Exception:
                try:
                    out = job.result()
                except Exception:
                    out = []
            return out if isinstance(out, (list, tuple)) else [out]

        if code in {"CANCELLED", "FAILED", "INTERRUPTED", "ERROR"}:
            raise SpaceRunError(f"Space job failed: {st}")

        if time.time() - start > timeout:
            raise SpaceRunError("The read/compute operation timed out.")

        if time.time() - last > 5:
            log.info("Space status: %s", code)
            last = time.time()
        time.sleep(2)

def _materialize_to_png(hf_space: str, candidate: str) -> str:
    """
    Take a candidate image (local path or data URL) and write a PNG into runs/.
    Falls back to copying the original if conversion fails.
    Returns the final local path.
    """
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    final_png = runs_dir / f"{Path(hf_space).name}_result.png"

    # data URL path (e.g., 'data:image/webp;base64,...')
    if candidate.lower().startswith("data:image/"):
        try:
            header, b64 = candidate.split(",", 1)
            raw = base64.b64decode(b64)
            from PIL import Image  # lazy import
            im = Image.open(io.BytesIO(raw))
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.save(str(final_png))
            return str(final_png)
        except Exception as e:
            raise SpaceRunError(f"Could not decode data URL image: {e}")

    # local file path
    src = Path(candidate)
    try:
        from PIL import Image  # lazy import
        im = Image.open(str(src))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(str(final_png))
        return str(final_png)
    except Exception as e:
        # Fallback: just copy with original extension so the UI can still serve it
        log.warning("PNG conversion failed (%s); copying original.", e)
        fallback = runs_dir / f"{Path(hf_space).name}_result{src.suffix.lower()}"
        shutil.copyfile(str(src), str(fallback))
        return str(fallback)

def call_space_flow(
    hf_space: str,
    image_path: str,
    calls: List[dict],
    timeout: int = 1200,
    hf_token: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    token = hf_token or os.getenv("HF_TOKEN")
    if token and not token.startswith("hf_"):
        raise SpaceRunError("HF_TOKEN looks invalid (should start with 'hf_').")
    client = Client(hf_space, hf_token=token)
    outputs: List[Any] = []

    for i, step in enumerate(calls, 1):
        api_name = step.get("api_name")
        raw_args = step.get("args", [])
        args = []
        for a in raw_args:
            if a == "{image}":
                args.append(handle_file(image_path))
            else:
                args.append(a)
        log.info("Calling %s step %d: %s args=%s", hf_space, i, api_name, raw_args)
        job = client.submit(*args, api_name=api_name)
        outputs = _wait_job(job, timeout)

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    out_any = _pick_image_like(outputs)
    if not out_any:
        raise SpaceRunError(f"No image-like output returned: {outputs}")

    final_path = _materialize_to_png(hf_space, out_any)
    return final_path, list(outputs)

# keep single-call helper for tools that only need one endpoint (optional)
def call_space_with_file(
    hf_space: str,
    image_path: str,
    api_name: Optional[str] = None,
    timeout: int = 1200,
    hf_token: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    # Default to /predict only for true single-endpoint Spaces
    out_path, outputs = call_space_flow(
        hf_space,
        image_path,
        calls=[{"api_name": api_name or "/predict", "args": ["{image}"]}],
        timeout=timeout,
        hf_token=hf_token,
    )
    return out_path, outputs
