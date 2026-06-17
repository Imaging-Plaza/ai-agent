# AI Imaging Agent (Imaging Plaza)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

AI Imaging Agent is a RAG + AI agent system that helps users discover the right imaging software for their images and tasks. Upload an image or volume, describe what you want to do, and get ranked tool recommendations with explanations and runnable demo links.

## Key Features

- **Conversational agent**: Multi-turn chat with tool search, alternatives, repository lookup, and demo follow-up actions.
- **React web app**: Modern chat UI with conversation history, queued follow-ups, dark/light theme, model selection, example prompts, and inline media.
- **FastAPI backend**: Authenticated API, SSE chat streaming, file upload/session handling, model discovery, health checks, and static SPA serving in production.
- **Smart retrieval**: Qwen3-Embedding-8B/BGE-style embeddings, FAISS search, and optional BGE reranking.
- **Vision-aware selection**: VLM reasoning over the user task, image previews, metadata, and retrieved catalog candidates.
- **Medical imaging focus**: DICOM, NIfTI, TIFF stacks, CT, MRI, microscopy, and other scientific imaging workflows.
- **Rich asset handling**: 2D previews, raw file serving, slice/MIP views, and browser-side 3D volume rendering for supported volumes.
- **Demo integration**: Links and guarded execution flows for runnable examples such as Hugging Face Gradio Spaces.

<p align="center">
    <img src="https://github.com/Imaging-Plaza/ai-agent/blob/develop/assets/example.gif?raw=true" height="700">
</p>

## Quick Start

### Prerequisites

- Python 3.10-3.12
- Node.js 20+ for frontend development
- OpenAI API key or an OpenAI-compatible endpoint key
- Internet access for model calls and optional catalog sync

### Install

```bash
git clone https://github.com/imaging-plaza/ai-agent.git
cd ai-agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[dev]"

cd src/frontend
npm install
cd ../..
```

The dev container workflow uses `uv`:

```bash
uv venv
uv pip install -e .
uv pip install -e ".[dev]"
```

### Configure

Create a `.env` file at the repository root:

```dotenv
# Agent model keys. Use the key referenced by config.yaml.
OPENAI_API_KEY=sk-xxxx
EPFL_API_KEY=your-epfl-key
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key

# Optional password for the React app. If unset, auth is disabled.
APP_PASSWORD=change-me

# Runtime paths
CONFIG_PATH=config.yaml
SOFTWARE_CATALOG=dataset/catalog.jsonl
RAG_INDEX_DIR=artifacts/rag_index

# Backend server
HOST=0.0.0.0
PORT=8000
DEV_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Recommendation defaults
TOP_K=8
NUM_CHOICES=3
EMBED_CATALOG_ON_START=1

# Optional catalog sync
GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
GRAPHDB_QUERY_FILE=src/ai_agent/queries/get_relevant_software.rq
SYNC_EVERY_HOURS=0

# Logging
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0
```

Models and retrieval backends are configured in `config.yaml`. The checked-in default uses an EPFL OpenAI-compatible endpoint for the default agent model and remote EPFL endpoints for embedding/reranking, while the UI model picker is populated from `available_models`.

### Run Locally

Start the FastAPI backend:

```bash
ai_agent serve
```

In a second terminal, start the React development server:

```bash
cd src/frontend
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api/*` to `http://localhost:8000`.

The legacy Gradio UI is still available:

```bash
ai_agent chat
```

### Build And Run The Production UI

```bash
cd src/frontend
npm run build
cd ../..
ai_agent serve
```

With a built frontend present, FastAPI serves both `/api/*` and the React app from the same origin.

### Docker

The root `Dockerfile` builds the React frontend and runs `ai_agent serve`.

```bash
docker build -t ai-agent .
docker run -p 7860:7860 --env-file .env ai-agent
```

Open `http://localhost:7860`.

## Usage

1. Sign in with `APP_PASSWORD` if auth is enabled.
2. Start from an example prompt or upload your own files.
3. Attach images, DICOM files, NIfTI volumes, TIFF stacks, videos, audio, or PDFs.
4. Describe the task, such as "segment the lungs from this CT scan".
5. Review ranked recommendations with accuracy scores, compatibility metadata, and explanations.
6. Use demo links or approve a pending demo action when the agent offers one.
7. Continue the conversation, ask for alternatives, or tune the model/top-k/number of choices from the header controls.

Supported slash embeds in the chat composer include:

```text
/help
/img <asset-id | name | url>
/audio <url>
/video <url>
/youtube <id | url>
/embed <url>
```

## Supported File Formats

- **Standard images**: PNG, JPG, JPEG, WEBP, BMP, GIF
- **Medical/scientific images**: DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`), TIFF/TIFF stacks
- **Other media**: MP3, WAV, MP4, MOV, WEBM, PDF, CSV, JSON, XML

Medical and volume files are converted into preview images for VLM analysis while their original format, dimensions, and metadata remain available for compatibility matching.

## Architecture

The system combines a React frontend, FastAPI service layer, conversational agent, and retrieval pipeline.

```text
React SPA
  - auth, chat, files, model picker, asset gallery, 3D volume view
        |
        v
FastAPI /api/*
  - /api/auth, /api/chat SSE, /api/files, /api/models, /api/catalog, /api/healthz
        |
        v
PydanticAI Agent
  - tool search, alternatives, repo info, demo actions
        |
        v
RAG Retrieval
  - metadata extraction, query construction, embeddings, FAISS, reranking
        |
        v
Software Catalog
  - JSONL catalog, optional GraphDB sync, runnable examples
```

Key source areas:

- `src/frontend/`: React + Vite web app
- `src/ai_agent/api/`: FastAPI app, routers, schemas, and pipeline dependencies
- `src/ai_agent/services/`: session, file, chat, and volume-view service logic
- `src/ai_agent/agent/`: PydanticAI agent and tools
- `src/ai_agent/retriever/`: embedding, FAISS, reranking, and catalog document loading
- `src/ai_agent/generator/`: prompts and structured response schemas
- `src/ai_agent/ui/`: legacy Gradio interface
- `src/ai_agent/catalog/`: optional GraphDB catalog synchronization

## CLI

```bash
ai_agent serve  # FastAPI backend; serves built React app when dist exists
ai_agent chat   # Legacy Gradio UI
ai_agent sync   # One-shot catalog sync and index refresh
```

Both `ai_agent` and `ai-agent` entry points are available.

## Development

```bash
# Python tests
pytest tests/

# Frontend checks
cd src/frontend
npm run lint
npm run build
```

When changing user-facing behavior, update the relevant docs pages under `docs/`. Keep `README.md`, `docs/index.md`, and `docs/guide.md` aligned.

## Documentation

Full documentation lives in `docs/` and is built with MkDocs Material:

```bash
mkdocs serve
```

Start with [docs/index.md](docs/index.md) or the maintainer-oriented [docs/guide.md](docs/guide.md).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## Credits

Developed by the Imaging Plaza team.

Core technologies include React, Vite, FastAPI, PydanticAI, OpenAI-compatible VLMs, FAISS, Qwen/BGE retrieval models, pydicom, nibabel, tifffile, and Three.js.

## Medical Disclaimer

This software recommends imaging tools. It is not a diagnostic tool and should not be used for clinical decisions without qualified medical review.
