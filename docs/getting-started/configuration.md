# Configuration

Configuration is split between environment variables in `.env` and model/retrieval settings in `config.yaml`.

## Minimal `.env`

Create `.env` in the repository root:

```dotenv
# Use the keys referenced by config.yaml.
OPENAI_API_KEY=sk-xxxx
EPFL_API_KEY=your-epfl-key
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key

CONFIG_PATH=config.yaml
SOFTWARE_CATALOG=dataset/catalog.jsonl
RAG_INDEX_DIR=artifacts/rag_index

# React/FastAPI auth. Leave unset to disable auth in local dev.
APP_PASSWORD=change-me

# FastAPI server defaults for ai_agent serve.
HOST=0.0.0.0
PORT=8000
DEV_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Recommendation defaults.
TOP_K=8
NUM_CHOICES=3
EMBED_CATALOG_ON_START=1

# Logging.
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0
```

## API Keys

The required key depends on `config.yaml`.

### EPFL Default

The checked-in default agent model uses the EPFL OpenAI-compatible endpoint:

```dotenv
EPFL_API_KEY=your-epfl-key
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key
```

`EPFL_API_KEY_EMBEDDER` is used by the remote embedder and reranker.

### Standard OpenAI

To use OpenAI directly, set an OpenAI model in `config.yaml` and provide:

```dotenv
OPENAI_API_KEY=sk-your-actual-key
```

## Model Configuration

`config.yaml` controls the default agent model, model picker entries, and retrieval stack.

```yaml
agent_model:
  name: "openai/gpt-oss-120b"
  base_url: "https://inference-rcp.epfl.ch/v1"
  api_key_env: "EPFL_API_KEY"

available_models:
  - display_name: "gpt-4o-mini"
    name: "gpt-4o-mini"
    base_url: null
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"

  - display_name: "openai/gpt-oss-120b [EPFL]"
    name: "openai/gpt-oss-120b"
    base_url: "https://inference-rcp.epfl.ch/v1"
    provider: "EPFL"
    api_key_env: "EPFL_API_KEY"

retrieval:
  embedder:
    backend: "remote"
    model_name: "Qwen/Qwen3-Embedding-8B"
    base_url: "https://inference-rcp.epfl.ch/v1"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
    timeout_s: 20

  reranker:
    backend: "remote"
    model_name: "BAAI/bge-reranker-v2-m3"
    base_url: "https://inference-rcp.epfl.ch/v1"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
    timeout_s: 20
```

The React model picker reads `/api/models`, which is populated from `available_models`.

## Running With OpenAI Models

```yaml
agent_model:
  name: "gpt-4o-mini"
  base_url: null
  api_key_env: "OPENAI_API_KEY"
```

Then set:

```dotenv
OPENAI_API_KEY=sk-your-actual-key
```

## Local Retrieval

To avoid remote embedding/reranking endpoints:

```yaml
retrieval:
  embedder:
    backend: "local"
    model_name: "BAAI/bge-m3"
  reranker:
    backend: "local"
    model_name: "BAAI/bge-reranker-v2-m3"
```

Local models require the relevant model downloads and enough CPU/GPU memory.

## Frontend And API Settings

### APP_PASSWORD

When set, the React app requires this shared passphrase and stores an httpOnly auth cookie after login.

```dotenv
APP_PASSWORD=change-me
```

When unset, auth is disabled. This is useful for local development behind trusted access controls.

### HOST / PORT

`ai_agent serve` reads:

```dotenv
HOST=0.0.0.0
PORT=8000
```

Docker overrides `PORT=7860`.

### DEV_CORS_ORIGINS

Allowed frontend origins for local Vite development:

```dotenv
DEV_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

In production, FastAPI serves the built frontend from the same origin, so CORS is normally irrelevant.

### FRONTEND_DIST_DIR

Path to the built React bundle:

```dotenv
FRONTEND_DIST_DIR=src/frontend/dist
```

If this directory contains `index.html`, `ai_agent serve` serves the SPA and static assets. If not, it assumes Vite is serving the frontend in development.

### VITE_API_TARGET

Used only by the frontend dev server proxy:

```bash
VITE_API_TARGET=http://localhost:8000 npm run dev
```

If unset, Vite proxies `/api/*` to `http://localhost:8000`.

## Catalog Sync

Manual and background catalog sync require a GraphDB SPARQL endpoint:

```dotenv
GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
GRAPHDB_QUERY_FILE=src/ai_agent/queries/get_relevant_software.rq
SYNC_EVERY_HOURS=24
```

Run one sync:

```bash
ai_agent sync
```

## Logging

```dotenv
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0
```

!!! warning
    `LOG_PROMPTS=1` can save prompt text and image previews locally. Use it only for debugging sessions.

## Verification

Check that the backend can start:

```bash
ai_agent serve
```

Check the frontend build:

```bash
cd src/frontend
npm run build
```

## Next Steps

- [Run the Quick Start](quickstart.md)
- Learn about [Using the Chat Interface](../user-guide/chat-interface.md)
- See the full [Environment Variables](../reference/environment.md) reference
