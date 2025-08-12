# ui/gradio_app.py (very top)
from __future__ import annotations

import os, sys
# add project root to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # <- override any prior config
)
log = logging.getLogger("ui")
log.info("Starting Gradio UI")

import tempfile
from pathlib import Path
from typing import Any, List, Optional

import gradio as gr
from PIL import Image

# Load env (OPENAI_API_KEY, HF_TOKEN, SOFTWARE_CATALOG, etc.)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# ---- Import your pipeline & models (from earlier messages/files) ----
from retriever.embedders import SoftwareDoc
from generator.schema import PerceptionCues
from api.pipeline import RAGImagingPipeline  # recommend_and_run()

CATALOG_PATH = os.getenv("SOFTWARE_CATALOG", "data/sample.jsonl")
WORKDIR = os.getenv("RAG_WORKDIR", "runs")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional, for private Spaces


def _load_catalog(path: str) -> list[SoftwareDoc]:
    import json
    try:
        import json5  # optional, for comments/single quotes
        parsers = (json.loads, json5.loads)
    except Exception:
        parsers = (json.loads,)

    docs: list[SoftwareDoc] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Software catalog not found at {p.resolve()}.\n"
            "Set SOFTWARE_CATALOG or create data/software_catalog.jsonl"
        )

    with p.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            last_err = None
            obj = None
            for parse in parsers:
                try:
                    obj = parse(line)
                    break
                except Exception as e:
                    last_err = e
            if obj is None:
                # show a short snippet to help debugging
                snippet = (line[:160] + "…") if len(line) > 160 else line
                raise ValueError(
                    f"Invalid catalog JSON on line {i}:\n{snippet}\n\nError: {last_err}"
                )
            # Pydantic v2 validate
            try:
                docs.append(SoftwareDoc.model_validate(obj))
            except Exception:
                docs.append(SoftwareDoc.parse_obj(obj))  # v1 fallback

    if not docs:
        raise ValueError("Catalog is empty.")
    return docs


# Build the pipeline once (cached across UI calls)
_pipe: Optional[RAGImagingPipeline] = None
def get_pipeline() -> RAGImagingPipeline:
    global _pipe
    if _pipe is None:
        docs = _load_catalog(CATALOG_PATH)
        _pipe = RAGImagingPipeline(docs, workdir=WORKDIR, hf_token=HF_TOKEN)
    return _pipe


def run_agent(task_text: str, image_file) -> tuple[Any, str, str]:
    """
    Gradio fn: takes free-form text + a file path, runs the RAG pipeline,
    returns (processed_image, chosen_software, rationale_or_error).
    """
    if not task_text or not image_file:
        return None, "", "Please provide both a task and an image."

    # Gradio File returns either a dict or a str (version-dependent)
    if isinstance(image_file, str):
        img_path = image_file
    elif isinstance(image_file, dict):
        img_path = image_file.get("name") or image_file.get("path")
    else:
        img_path = None

    if not img_path or not os.path.exists(img_path):
        return None, "", "Could not read the uploaded file path."

    log.info("run_agent: task='%s' file='%s'", task_text, img_path)

    pipe = get_pipeline()

    # Optional light cue from text only (we keep UI minimal: text + file)
    cues = PerceptionCues(
        task=("segmentation" if "segment" in task_text.lower() else None)
    )

    result = pipe.recommend_and_run(image_path=img_path, user_task=task_text, cues=cues)

    if "error" in result:
        return None, "", f"❌ {result['error']}\n\nDetails: {result.get('errors','')}"

    # The pipeline copies the Space output to a PNG in WORKDIR; display it.
    out_img_path = result["result_image"]
    try:
        from PIL import Image
        out_img = Image.open(out_img_path).convert("RGB")
    except Exception as e:
        log.warning("Could not open result image '%s': %s", out_img_path, e)
        out_img = None

    chosen = result["choice"]
    why = result.get("why", result.get("generator", {}).get("why", "")) or "(no rationale provided)"
    return out_img, f"Selected software: {chosen}", why



# ------------- Gradio UI (text + image only) -------------
with gr.Blocks(title="AI Imaging Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🔎 AI-assisted software picker\n"
                "Describe your task and drop an image. "
                "The agent will select the right software from our catalog, run its Hugging Face Space, "
                "and return the processed image.")

    with gr.Row():
        task_box = gr.Textbox(label="Describe your task", placeholder="e.g., \"Segment the lungs\" or \"Deblur this photo\"", lines=2)
    image_in = gr.File(
            label="Image (drag & drop)",
            file_count="single",
            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nii", ".nii.gz"]
        )

    run_btn = gr.Button("Run", variant="primary")

    out_image = gr.Image(label="Result", interactive=False)
    out_choice = gr.Markdown()
    out_why = gr.Markdown()

    run_btn.click(fn=run_agent, inputs=[task_box, image_in], outputs=[out_image, out_choice, out_why])

# For local testing; when embedding into your website, import `demo` and mount.
if __name__ == "__main__":
    demo.queue(api_open=False).launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        inbrowser=True,
        show_error=True,   # <— useful while debugging
    )
