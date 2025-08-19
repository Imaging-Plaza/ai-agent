# AI Imaging Agent — RAG + VLM Tool Picker

A tiny “AI-assisted search” that helps users find the **right imaging software** for their image and task.  
Users drop an image and describe what they want (e.g., *“segment the lungs”*). The app:

1. **Retrieves** candidate tools from a local catalog (text-only query + format token).
2. **Selects** the best tool with a **single VLM call** (text + image + candidates + original image metadata).
3. **Returns a link** to the tool’s **public runnable demo** (Hugging Face Space, notebook viewer, etc.).  
   *(We don’t run the tool or upload user data to third-party endpoints.)*

---

## What’s in here

- **Retrieval**: FAISS/BGE-M3 + Cross-Encoder reranker.
- **Single-shot selection**: OpenAI VLM (`gpt-4o`/`gpt-4o-mini` by default).
- **Image metadata awareness**: The original file extension & shape are passed to the VLM (even if a `.tif`/`.nii.gz` is rasterized to PNG for the preview), so IO compatibility matters in the choice.
- **Gradio UI**: One textbox + one file input → result = selected software + demo link.
- **Logging**: Console + rotating file logs; optional **prompt snapshots** to `logs/`.

> ⚠️ **Medical disclaimer**: This app is a software recommender, not a diagnostic tool.

---

## Quickstart

### 1) Requirements
- Python 3.10–3.12
- A working internet connection (for model calls)
- An OpenAI API key

```bash
git clone <your-repo>
cd ai-agent
python -m venv env
# Windows
env\Scripts\activate
# macOS/Linux
source env/bin/activate

pip install -r requirements.txt
```

### 2) Configure `.env`

Create a `.env` file at repo root:

```dotenv
OPENAI_API_KEY=sk-xxxx
# Optional model overrides (defaults work):
OPENAI_VLM_MODEL=gpt-4o
OPENAI_MODEL=gpt-4o-mini

# Software catalog
SOFTWARE_CATALOG=path/to/your/catalog.jsonl

# Logging
LOGLEVEL_CONSOLE=WARNING     # INFO or WARNING
LOGLEVEL_FILE=INFO       # INFO or DEBUG
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=1            # write selector prompt snapshots (text + the PNG preview the VLM saw)

# VLM usage policy / confidence gate
FORCE_VLM=0              # 1 = always call VLM; 0 = use reranker confidence gate
RERANK_MARGIN=0.15       # if (top - second) > margin OR top >= RERANK_TOP, skip VLM
RERANK_TOP=0.90
```

### 3) Run the app

```bash
python -m ui.gradio_app
# or
python ui/gradio_app.py
```

Open http://127.0.0.1:7860 and try:  
> “I want to segment the lungs from this CT scan image” + a `.tif` lung volume slice (or any image).

---

## Catalog format

The catalog is JSON **or** JSONL. Each line/object is a **SoftwareDoc**. Minimal fields:

```json
{
  "name": "3d-lungs-segmentation",
  "tasks": ["segmentation"],
  "modality": ["CT"],
  "dims": ["3D"],
  "anatomy": ["lung"],
  "input_formats": ["TIFF","TIF"],
  "output_types": ["mask"],
  "language": "Python",
  "gpu_required": false,
  "license": "Apache-2.0",

  "description": "3D lung segmentation from CT; returns a mask/overlay.",

  // Runnable entry points (optional; first by priority is used for the "Open demo" link)
  "runnables": [
    { "type": "gradio", "url": "https://huggingface.co/spaces/qchapp/3d-lungs-segmentation", "priority": 10 }
  ],

  // Back-compat: if you still use hf_space only
  "hf_space": "qchapp/3d-lungs-segmentation"
}
```

> You can add multiple runnable types (e.g., `"type": "notebook"`, `"type": "webapp"`, `"type": "jvm"`); the pipeline just picks the **best base URL** to show to the user.

---

## How the pipeline works

### Retrieval (fast, no LLM)
- Build a text query from the user prompt.
- If the user uploaded a file, add a **format token** (e.g., `format:TIF` or `format:NII.GZ`) to bias toward IO-compatible tools.
- Embed with **BGE-M3**, rerank with a Cross-Encoder, take top-K.

### Selection (one VLM call)
- If retrieval is **very confident** (top score >> second), pick top-1.
- Otherwise, call the **VLM** with:
  - **Text**: user request + compact table of top-K candidates.
  - **Image**: a **PNG preview** (safe for the API).
  - **Metadata**: the **original** file name, extension, frames/shape/dtype/zooms (for NIfTI), etc.
- VLM responds with **strict JSON**:
  ```json
  {"choice":"<name>","alternates":["..."],"why":"..."}
  ```
- UI displays the choice and a **public runnable link** (no data upload, no server-side execution).

---

## Logging & Debugging

- **Console log level** via `LOGLEVEL_CONSOLE` (default `INFO`).
- **File logs** in `logs/app_YYYYMMDD.log` (enable with `FILE_LOG=1`).
- **Prompt snapshots** (when `LOG_PROMPTS=1`):
  - `logs/vlm_selector_YYYYMMDD_HHMMSS.txt` — the system/user text the model saw
  - `logs/vlm_selector_YYYYMMDD_HHMMSS.png` — the exact PNG sent to the VLM

---

## Security & Privacy

- The app does **not** upload your image to third-party demos.  
  It only shows a **link** to a public demo page.
- The only external API call on your content is the **single VLM request** to OpenAI for tool selection (preview image + brief metadata + text).  
  Turn off prompt snapshots if you don’t want local copies of previews: `LOG_PROMPTS=0`.

---

## Project layout

```
api/
  pipeline.py         # retrieval + selection + link-out
generator/
  generator.py        # VLMToolSelector (single-call selection)
  prompts.py          # SELECTOR_SYSTEM (strict JSON)
  schema.py           # schema of the prompts
retriever/
  embedders.py        # BGE-M3 + FAISS + CrossEncoder
ui/
  gradio_app.py       # minimal UI
utils/
  image_meta.py       # summarize_image_metadata(), detect_ext_token()
  image_analyzer.py   # convert image to png to send it through gpt
data/
  sample.jsonl        # example catalog
logs/
  ...                 # rotating app logs + prompt snapshots
```

---

## Development tips

- Run UI from project root:  
  `python -m ui.gradio_app`
- Toggle **always-VLM** selection during demos:  
  `FORCE_VLM=1`
- Save selector prompts to compare changes:  
  `LOG_PROMPTS=1`

---

## Future improvements

- [ ] **Deep links**: detect Space endpoints that accept `file_url` and assemble a *safe* one-click URL when explicitly allowed by the maintainer.
- [ ] **Notebook/JVM runnable examples**: add known viewer/launcher patterns (e.g., nbviewer URLs, custom JVM launch pages) to `runnables`.
- [ ] **Multi-image/volume support**: accept stacks, 3D/4D previews, and summarize more DICOM/NIfTI metadata.
- [ ] **Additional VLM providers**: add Anthropic/Google or local VLMs behind a simple provider interface.
- [ ] **Index persistence**: save FAISS index to disk; incremental updates.
- [ ] **Reranker finetuning**: train a domain reranker on curated (query, tool) pairs.
- [ ] **Evaluation harness**: small benchmark set (queries + ground-truth tool) with metrics.
- [ ] **Catalog ingestion**: adding all Imaging Plaza softwares to the catalog.
- [ ] **UI extras**: history, “copy link”, toolcards with tags, lightweight analytics (opt-in).
- [ ] **Containerization**: Dockerfile + Compose for easy deploy.
- [ ] **Testing/CI**: unit tests for metadata, prompt builders, and link selectors.

---

## License

No license for now.

---
