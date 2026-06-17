# Environment Variables

Configuration is loaded from `.env` by `python-dotenv` when the application starts.

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Conditional | unset | OpenAI API key when `config.yaml` uses an OpenAI model |
| `EPFL_API_KEY` | Conditional | unset | EPFL OpenAI-compatible endpoint key for agent models |
| `EPFL_API_KEY_EMBEDDER` | Conditional | unset | EPFL endpoint key for remote embedder/reranker |
| `APP_PASSWORD` | No | unset | Shared passphrase for React/FastAPI auth; auth disabled when unset |
| `HOST` | No | `0.0.0.0` | Host used by `ai_agent serve` |
| `PORT` | No | `8000` | Port used by `ai_agent serve`; Docker sets `7860` |
| `DEV_CORS_ORIGINS` | No | `http://localhost:5173,http://127.0.0.1:5173` | Allowed Vite dev origins |
| `FRONTEND_DIST_DIR` | No | `src/frontend/dist` | Built React bundle served by FastAPI when present |
| `UPLOAD_ROOT` | No | system temp `ai_agent_uploads` | Upload storage root for session assets |
| `CONFIG_PATH` | No | `config.yaml` | YAML model/retrieval configuration path |
| `SOFTWARE_CATALOG` | No | `dataset/catalog.jsonl` | Software catalog JSONL path |
| `RAG_INDEX_DIR` | No | `artifacts/rag_index` | FAISS index artifact directory |
| `TOP_K` | No | `8` | Default number of retrieval candidates |
| `NUM_CHOICES` | No | `3` | Default number of final recommendations |
| `EMBED_CATALOG_ON_START` | No | `1` | Build/embed catalog at startup when needed |
| `AGENT_OUTPUT_RETRIES` | No | `3` | Structured-output retry count |
| `AGENT_CACHE_MAX` | No | `16` | Max cached agent instances |
| `IMAGE_META_CACHE_MAX` | No | `128` | Max image metadata cache entries |
| `SYNC_EVERY_HOURS` | No | `0` | Background catalog refresh interval |
| `SYNC_SKIP_IF_FRESH_SECONDS` | No | `0` | Skip sync when local catalog is fresh |
| `SYNC_FORCE` | No | `0` | Force catalog sync |
| `GRAPHDB_URL` | For sync | unset | GraphDB SPARQL endpoint |
| `GRAPHDB_GRAPH` | For sync | unset | Named graph IRI |
| `GRAPHDB_QUERY_FILE` | For sync | `get_relevant_software.rq` | SPARQL query file |
| `GRAPHDB_USER` | No | unset | GraphDB username |
| `GRAPHDB_PASSWORD` | No | unset | GraphDB password |
| `OUTPUT_JSONLD` | No | `dataset/catalog.jsonld` | Raw JSON-LD sync output |
| `OUTPUT_JSONL` | No | `dataset/catalog.jsonl` | Processed JSONL sync output |
| `GITHUB_TOKEN` | No | unset | GitHub token for repository info lookup |
| `LOGLEVEL_CONSOLE` | No | `WARNING` | Console log level |
| `LOGLEVEL_FILE` | No | `INFO` | File log level |
| `FILE_LOG` | No | `1` | Enable file logging |
| `LOG_DIR` | No | `logs` | Log directory |
| `LOG_PROMPTS` | No | `0` | Save prompt snapshots for debugging |
| `DEBUG` | No | `0` | Enable debug mode |

!!! note "Frontend dev variable"
    `VITE_API_TARGET` is read by Vite, not Python. It defaults to `http://localhost:8000` and controls where `/api/*` is proxied during `npm run dev`.

## API Keys

### `OPENAI_API_KEY`

Required when `config.yaml` points an agent model or model picker option at the standard OpenAI endpoint.

```dotenv
OPENAI_API_KEY=sk-xxxx
```

### `EPFL_API_KEY`

Required when `config.yaml` uses the EPFL OpenAI-compatible inference endpoint for the agent model.

```dotenv
EPFL_API_KEY=your-epfl-key
```

### `EPFL_API_KEY_EMBEDDER`

Required for the default remote embedder and reranker configuration.

```dotenv
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key
```

If the key is missing, the remote reranker may be disabled or fail depending on the selected retrieval backend.

### `GITHUB_TOKEN`

Optional token used by repository information tools.

```dotenv
GITHUB_TOKEN=ghp_xxxx
```

## React/FastAPI Runtime

### `APP_PASSWORD`

When set, `/api/auth/login` requires this shared passphrase and sets an httpOnly cookie.

```dotenv
APP_PASSWORD=change-me
```

When unset, auth is disabled. Use this only in trusted local development or behind another access-control layer.

### `HOST` / `PORT`

Used by `ai_agent serve`.

```dotenv
HOST=0.0.0.0
PORT=8000
```

The Docker image sets `PORT=7860`.

### `DEV_CORS_ORIGINS`

Comma-separated list of frontend dev origins allowed to call the backend:

```dotenv
DEV_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

### `FRONTEND_DIST_DIR`

Directory containing the built React app:

```dotenv
FRONTEND_DIST_DIR=src/frontend/dist
```

If `index.html` exists there, FastAPI serves the SPA. Otherwise it assumes Vite is running separately.

### `UPLOAD_ROOT`

Root directory for uploaded files grouped by session.

```dotenv
UPLOAD_ROOT=/tmp/ai_agent_uploads
```

## Model Configuration

### `CONFIG_PATH`

```dotenv
CONFIG_PATH=config.yaml
```

This file defines `agent_model`, `available_models`, and retrieval backend settings.

## Pipeline

### `SOFTWARE_CATALOG`

```dotenv
SOFTWARE_CATALOG=dataset/catalog.jsonl
```

### `RAG_INDEX_DIR`

```dotenv
RAG_INDEX_DIR=artifacts/rag_index
```

### `TOP_K` / `NUM_CHOICES`

```dotenv
TOP_K=8
NUM_CHOICES=3
```

The React header can override these values per request.

### `EMBED_CATALOG_ON_START`

```dotenv
EMBED_CATALOG_ON_START=1
```

Set to `0` when you know the index already exists and want faster cold starts.

## Catalog Sync

These variables control `ai_agent sync` and background refresh.

```dotenv
GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
GRAPHDB_QUERY_FILE=src/ai_agent/queries/get_relevant_software.rq
OUTPUT_JSONLD=dataset/catalog.jsonld
OUTPUT_JSONL=dataset/catalog.jsonl
SYNC_EVERY_HOURS=24
SYNC_SKIP_IF_FRESH_SECONDS=0
SYNC_FORCE=0
```

Authenticated GraphDB endpoints can also use:

```dotenv
GRAPHDB_USER=myuser
GRAPHDB_PASSWORD=mypassword
```

## Logging

```dotenv
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0
DEBUG=0
```

!!! warning
    `LOG_PROMPTS=1` may save prompt text and image previews under `logs/`. Do not enable it for sensitive data unless that is acceptable.

## Complete Example

```dotenv
OPENAI_API_KEY=sk-xxxx
EPFL_API_KEY=your-epfl-key
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key
GITHUB_TOKEN=ghp_xxxx

APP_PASSWORD=change-me
HOST=0.0.0.0
PORT=8000
DEV_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
FRONTEND_DIST_DIR=src/frontend/dist

CONFIG_PATH=config.yaml
SOFTWARE_CATALOG=dataset/catalog.jsonl
RAG_INDEX_DIR=artifacts/rag_index

TOP_K=8
NUM_CHOICES=3
EMBED_CATALOG_ON_START=1
AGENT_OUTPUT_RETRIES=3
AGENT_CACHE_MAX=16
IMAGE_META_CACHE_MAX=128

GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
GRAPHDB_QUERY_FILE=src/ai_agent/queries/get_relevant_software.rq
SYNC_EVERY_HOURS=0
SYNC_SKIP_IF_FRESH_SECONDS=0
SYNC_FORCE=0

LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0
DEBUG=0
```

## Next Steps

- [Configuration Guide](../getting-started/configuration.md)
- [CLI Commands](cli.md)
- [Architecture Overview](../architecture/overview.md)
