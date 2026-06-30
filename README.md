# AI Imaging Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

AI Imaging Agent, also called Imaging Plaza, is a RAG plus agent system for finding and running imaging software. A user can upload images or scientific/medical files, describe a task, receive ranked software recommendations from a catalog, and optionally execute configured Gradio demos on the uploaded data.

The current application has two UI surfaces:

- A React single-page app backed by FastAPI, started with `ai_agent serve`.
- A legacy Gradio chat UI, started with `ai_agent chat`.

## What It Does

- Recommends imaging tools from a catalog using semantic retrieval and reranking.
- Uses an agent/VLM step to reason over the user request, image preview, metadata, and retrieved candidates.
- Supports multi-turn chat, file uploads, conversation history, model selection, and streamed FastAPI chat events.
- Handles common image, medical, scientific, video, audio, and PDF uploads.
- Can import runnable Hugging Face Gradio Spaces from links and map catalog recommendations to executable endpoints.
- Can ask for user approval and runtime parameters before executing a configured tool.
- Can run simple compatible tool chains when configured endpoint contracts allow it.

## Repository Layout

```text
src/ai_agent/
  agent/        PydanticAI agent, prompts-to-tools orchestration, MCP-style tools
  api/          FastAPI server, routers, schemas, shared pipeline dependencies
  catalog/      GraphDB catalog sync and FAISS index refresh
  config/       Packaged default Gradio tool registry
  core/         Shared pipeline registry
  generator/    Selection prompts and structured recommendation schemas
  retriever/    Embeddings, FAISS index, reranker, SoftwareDoc catalog model
  services/     Transport-independent chat, sessions, files, workflows
  ui/           Legacy Gradio interface
  utils/        Config, validation, image IO, previews, metadata, cache, cleanup

src/frontend/   React/Vite frontend for the FastAPI app
tests/          Pytest suite
docs/           MkDocs documentation
data/           Sample catalog/query assets
tools/          Container and deployment helpers
```

## Requirements

- Python 3.10 or newer.
- Node.js 20 or newer for the React frontend.
- An API key for the configured agent model.
- An API key for the configured remote embedder/reranker, unless using local retrieval models.

The repository is configured for editable Python installs through `pyproject.toml`.

## Install

Recommended inside the dev container:

```bash
uv venv
uv pip install -e .
uv pip install -e ".[dev]"
```

Local pip alternative:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate a local environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install frontend dependencies when working on the React app:

```bash
cd src/frontend
npm install
```

## Configuration

Runtime configuration is loaded from environment variables and `config.yaml`.

The checked-in `config.yaml` currently uses EPFL's OpenAI-compatible endpoint for the default agent model:

```yaml
agent_model:
  name: "openai/gpt-oss-120b"
  base_url: "https://inference-rcp.epfl.ch/v1"
  api_key_env: "EPFL_API_KEY"
```

It also exposes OpenAI models in the frontend model picker and configures remote EPFL retrieval services:

```yaml
retrieval:
  embedder:
    backend: "remote"
    model_name: "Qwen/Qwen3-Embedding-8B"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
  reranker:
    backend: "remote"
    model_name: "BAAI/bge-reranker-v2-m3"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
```

Create a local `.env` file or export equivalent variables. Do not commit secrets.

```dotenv
# Agent/model access
EPFL_API_KEY=...
EPFL_API_KEY_EMBEDDER=...
OPENAI_API_KEY=...
CONFIG_PATH=config.yaml

# Catalog and index
SOFTWARE_CATALOG=dataset/catalog.jsonl
RAG_INDEX_DIR=artifacts/rag_index
EMBED_CATALOG_ON_START=1
TOP_K=8
NUM_CHOICES=3

# FastAPI / frontend
HOST=0.0.0.0
PORT=8000
APP_PASSWORD=
DEV_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Gradio tool registry
AI_AGENT_GRADIO_TOOLS_CONFIG=src/ai_agent/config/gradio_tools.json

# Logging and cleanup
FILE_LOG=1
LOG_DIR=logs
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
LOG_PROMPTS=0
CACHE_DB_PATH=
CLEANUP_INTERVAL_SECONDS=7200
LOG_RETENTION_DAYS=7
```

If `APP_PASSWORD` is unset, FastAPI auth is disabled for local development. If it is set, the frontend login page stores an HTTP-only cookie after successful login.

## Run The App

### React plus FastAPI development mode

Start the backend:

```bash
ai_agent serve
```

By default this serves FastAPI on `http://localhost:8000` and exposes API docs at `http://localhost:8000/api/docs`.

Start the frontend in a second terminal:

```bash
cd src/frontend
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api/*` to the backend.

### Production-style local run

Build the frontend and let FastAPI serve the compiled bundle:

```bash
cd src/frontend
npm run build
cd ../..
PORT=7860 ai_agent serve
```

Open `http://localhost:7860`.

### Legacy Gradio UI

```bash
ai_agent chat
```

This launches the older Gradio interface. It shares the same pipeline and catalog index code but is not the primary React/FastAPI surface.

## CLI

The installed package provides both `ai_agent` and `ai-agent` entry points.

```bash
ai_agent serve   # Start FastAPI backend; serves React dist if it exists
ai_agent chat    # Start legacy Gradio chat UI
ai_agent sync    # Sync catalog from GraphDB and update the FAISS index
```

`ai_agent sync` expects GraphDB-related configuration when pulling a remote catalog:

```dotenv
GRAPHDB_URL=https://...
GRAPHDB_GRAPH=https://...
GRAPHDB_QUERY_FILE=src/ai_agent/queries/get_relevant_software.rq
GRAPHDB_USER=
GRAPHDB_PASSWORD=
OUTPUT_JSONLD=dataset/catalog.jsonld
OUTPUT_JSONL=dataset/catalog.jsonl
SYNC_EVERY_HOURS=0
SYNC_SKIP_IF_FRESH_SECONDS=0
SYNC_FORCE=0
```

## Catalog And Retrieval

The recommender reads software records as JSONL or a JSON array. The default path is `dataset/catalog.jsonl`, configurable with `SOFTWARE_CATALOG`.

Each record is parsed into `SoftwareDoc`, which tolerates extra catalog fields and normalizes common schema.org-style names such as:

- `name`
- `description`
- `applicationCategory`
- `featureList`
- `imagingModality`
- `keywords`
- `dims`
- `anatomy`
- `supportingData`
- `runnableExample`
- `hasExecutableNotebook`
- `codeRepository`
- `license`

Retrieval works in stages:

1. The user message is stripped of control tags such as `[EXCLUDE:tool1|tool2]`.
2. Uploaded files contribute metadata hints, including format tokens such as `format:dicom`, `format:nifti`, or `format:tiff`.
3. The query is embedded and searched against a FAISS index.
4. A CrossEncoder-style reranker sorts the candidate pool.
5. The agent receives the request, image preview bytes when available, metadata, conversation history, and retrieved candidates.

If the FAISS index is empty and `EMBED_CATALOG_ON_START=1`, startup tries to embed the configured catalog into `artifacts/rag_index`.

## Supported Uploads

The file validator accepts:

- Images: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.webp`, `.bmp`, `.gif`
- Medical/scientific: `.dcm`, `.nii`, `.nii.gz`, DICOM directories, ZIP archives containing DICOM files
- Video: `.mp4`, `.mov`, `.webm`, `.mkv`
- Audio: `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a`
- Documents: `.pdf`

Image previews and metadata are generated for agent reasoning where possible. If a preview cannot be built, the system falls back to text and metadata paths instead of failing the whole turn.

## Runnable Gradio Tools

Configured runnable tools live in `src/ai_agent/config/gradio_tools.json` by default. Override the path with:

```dotenv
AI_AGENT_GRADIO_TOOLS_CONFIG=/path/to/gradio_tools.json
```

The registry models Gradio applications separately from callable endpoints, so one Hugging Face Space can expose multiple operations. Tool and endpoint entries can define:

- catalog aliases used to match recommendations
- Gradio API endpoint names
- input mappings from session files, descriptions, literal values, or runtime parameters
- output selectors for previews, downloads, metadata, success, and errors
- approval settings
- optional input/output contracts for workflow planning

In the React app, open `/tools` to add a Hugging Face Space link such as:

```text
user-tool.hf.space
huggingface.co/user/tool
huggingface.co/spaces/user/tool
```

The backend reads the Space metadata from Gradio's `/gradio_api/info` and `/gradio_api/mcp/schema` endpoints, saves the generated registry entry, and reloads the tool registry.

## API Surface

FastAPI mounts application routes under `/api/*`.

- `GET /api/healthz`
- `GET /api/models`
- `POST /api/chat`
- `POST /api/chat/{session_id}/approve`
- `POST /api/chat/{session_id}/decline`
- `POST /api/files`
- `GET /api/files/preview/{asset_id}`
- `GET /api/files/asset/{asset_id}/raw`
- `GET /api/files/asset/{asset_id}/view`
- `GET /api/files/asset/{asset_id}/volume`
- `GET/POST /api/gradio-tools...`
- `GET/POST /api/auth...`

Chat responses are streamed as server-sent events. Events include `session`, `status`, `text`, `recommendation`, `tool_trace`, `pending_action`, `clarification`, `images`, `files`, `usage`, `error`, and `done`.

## Development

Run tests:

```bash
pytest tests/
```

Run frontend checks:

```bash
cd src/frontend
npm run lint
npm run build
```

Run Python formatting/linting tools when relevant:

```bash
ruff check src tests
black --check src tests
mypy src
```

The repository currently contains a `justfile`, but its `serve` tasks still refer to the removed `ai_agent ui` command. Use the real CLI modes above.

## Docker

The root `Dockerfile` builds the React frontend, installs the Python package, and starts:

```bash
ai_agent serve
```

Build and run manually:

```bash
docker build -t ai-agent:dev .
docker run --env-file .env -p 7860:7860 ai-agent:dev
```

`docker-compose.yml` runs the app on port `7860` inside the container and includes a Cloudflare tunnel sidecar.

## Documentation

Additional documentation lives under `docs/`:

- `docs/guide.md` for contributor-oriented repository guidance
- `docs/architecture/overview.md` for architecture notes
- `docs/getting-started/` for setup and configuration pages
- `docs/user-guide/` for user workflows
- `docs/reference/` for CLI and environment reference

Some docs may lag behind the current React/FastAPI implementation; when behavior conflicts, prefer the executable code in `src/ai_agent/` and the CLI in `src/ai_agent/cli.py`.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Medical Disclaimer

AI Imaging Agent recommends software tools. It is not a diagnostic system and should not be used as a substitute for qualified medical judgment.
