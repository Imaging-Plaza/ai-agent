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

# Install with pip using pyproject.toml
pip install --upgrade pip
pip install .

# For development (includes test dependencies)
pip install -e ".[dev]"
```

### 2) Configure `.env`

Create a `.env` file at repo root:

```dotenv
OPENAI_API_KEY=sk-xxxx
# Optional model overrides (defaults work):
OPENAI_MODEL=gpt-4o

# Software catalog
SOFTWARE_CATALOG=path/to/your/catalog.jsonl

# Pipeline configuration
TOP_K=8                # Number of candidates to retrieve
NUM_CHOICES=3          # Number of tools to recommend

# Logging configuration
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0         # write selector prompt snapshots
```

### 3) Run the app

```bash
ai_agent ui
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
- Build a text query from the user prompt
- If user uploaded a file, add a **format token** (e.g., `format:TIF` or `format:NII.GZ`)
- Embed with **BGE-M3**, rerank with Cross-Encoder
- Return top-K candidates (configurable via `TOP_K`)

### Selection (one VLM call)
- Call the **VLM** with:
  - **Text**: user request + compact table of top-K candidates
  - **Image**: a **PNG preview** (safe for the API)
  - **Metadata**: original file info (name, extension, shape, etc.)
- VLM responds with **strict JSON**:
  ```json
  {
    "choices": [
      {
        "name": "tool-name",
        "rank": 1,
        "accuracy": 95.5,
        "why": "Best match because..."
      },
      {
        "name": "alternative-tool",
        "rank": 2,
        "accuracy": 82.3,
        "why": "Good alternative..."
      }
    ]
  }
  ```

- Returns up to `NUM_CHOICES` ranked tools with accuracy scores
- UI displays choices with explanation and demo links

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
ai_agent/
  api/
    pipeline.py       # RAG pipeline implementation
  generator/
    generator.py      # VLMToolSelector implementation
    prompts.py       # System prompts and templates
    schema.py        # Pydantic models for validation
  retriever/
    software_doc.py  # Catalog document schema
    text_embedder.py # Sentence embedding models
    reranker.py      # Cross-encoder reranker
    vector_index.py  # FAISS index management
  ui/
    app.py           # Gradio interface
  utils/
    file_validator.py  # File format validation
    image_meta.py     # Metadata extraction
    image_io.py       # Image loading/conversion
    image_analyzer.py # VLM image analysis
    tags.py           # Tags passed to the VLM
    previews.py       # Building previews for user
tests/               # Unit tests
pyproject.toml       # Project configuration and dependencies
```

---

## Docker deployment

You can find the docker image in `tools/image/Dockerfile`

### Build and run - app starts automatically

```bash
docker build -t ai-agent:latest -f tools/image/Dockerfile .
docker run -d --rm -p 7860:7860 ai-agent:latest
```

### With environment variables

```bash
docker run -d --rm -p 7860:7860 \
  -e OPENAI_API_KEY="your-key" \
  ai-agent:latest
```

---

## Development tips

- Run UI from project root:  
  `python -m ui.gradio_app`
- Save selector prompts to compare changes:  
  `LOG_PROMPTS=1`
- Use a devcontainer when working with `vscode`
  `.devcontainer/devcontainer.json`

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
