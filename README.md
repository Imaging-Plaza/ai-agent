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
  "description": "3D lung segmentation from CT; returns a mask/overlay.",

  "applicationCategory": [],
  "featureList": ["segmentation"],
  "imagingModality": ["CT"],
  "dims": [3],
  "anatomy": ["lung"],
  "keywords": ["mask", "overlay", "lung segmentation", "CT"],

  "programmingLanguage": "Python",
  "requiresGPU": false,
  "isAccessibleForFree": true,
  "license": "Apache-2.0",

  "supportingData": [
    {
      "datasetFormat": "TIFF",
      "bodySite": "lung",
      "imagingModality": "CT",
      "hasDimensionality": 3
    },
    {
      "datasetFormat": "TIF",
      "bodySite": "lung",
      "imagingModality": "CT",
      "hasDimensionality": 3
    }
  ],

  "runnableExample": [
    {
      "hostType": "gradio",
      "url": "https://huggingface.co/spaces/qchapp/3d-lungs-segmentation",
      "name": "HF Space"
    }
  ]
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
scripts/
  ...                 # some preprocessing scripts for the dataset curation
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

- [x] **Notebook / JVM runnable examples** — support known viewer/launcher patterns (e.g., nbviewer URLs, custom JVM launch pages) in `runnables`.
- [x] **Multi-image / volume support** — accept stacks, generate 3D/4D previews, and summarize richer DICOM / NIfTI metadata.
- [x] **Index persistence** — save the FAISS index to disk with support for incremental updates.
- [x] **Catalog ingestion** — integrate all Imaging Plaza software entries into the catalog.

- [ ] **Deep links** — detect Space endpoints that accept `file_url` and safely assemble one-click URLs when explicitly allowed by the maintainer.
- [ ] **Additional VLM providers** — add Anthropic / Google APIs or local open-source VLMs behind a simple provider interface.
- [ ] **Reranker fine-tuning** — train a domain-specific CrossEncoder on curated (query, tool) pairs and optionally distill to a lighter model for speed.
- [ ] **Evaluation harness** — provide a small benchmark set (queries + expected tools) with metrics such as top-1 accuracy, MRR, and score margins.
- [ ] **UI enhancements** — add result history, “copy demo link” buttons, compact toolcards with tags/modality/license, and lightweight opt-in analytics.
- [ ] **Containerization** — supply Dockerfile and docker-compose for reproducible deployment, with GPU acceleration support if available.
- [ ] **Testing / CI** — expand unit tests (metadata parsing, preview builders, link selectors) and configure GitHub Actions for linting, tests, and catalog validation.


---

## License

No license for now.

---
