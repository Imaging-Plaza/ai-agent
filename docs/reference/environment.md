# Environment Variables

Configuration for the AI Imaging Agent is managed via environment variables, typically defined in a `.env` file in the repository root. The application loads this file automatically on startup using `python-dotenv`.

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Conditional | — | OpenAI API key (if using OpenAI endpoint) |
| `EPFL_API_KEY` | Conditional | — | API key for the EPFL inference endpoint (agent model) |
| `EPFL_API_KEY_EMBEDDER` | Conditional | — | API key for EPFL embedder/reranker endpoint |
| `CONFIG_PATH` | No | `config.yaml` | Path to YAML model/retrieval configuration |
| `SOFTWARE_CATALOG` | No | `dataset/catalog.jsonl` | Path to software catalog JSONL file |
| `RAG_INDEX_DIR` | No | `artifacts/rag_index` | Directory for FAISS index artifacts |
| `NUM_CHOICES` | No | `3` | Default number of tool recommendations |
| `EMBED_CATALOG_ON_START` | No | `1` (enabled) | Embed catalog at pipeline startup |
| `AGENT_OUTPUT_RETRIES` | No | `3` | Agent structured-output retry count |
| `AGENT_CACHE_MAX` | No | `16` | Max cached agent instances (LRU) |
| `SYNC_EVERY_HOURS` | No | `0` (disabled) | Background catalog refresh interval |
| `SYNC_SKIP_IF_FRESH_SECONDS` | No | `0` (disabled) | Skip remote sync if local catalog is fresh |
| `SYNC_FORCE` | No | `0` | Force sync regardless of freshness |
| `GRAPHDB_URL` | For sync | — | GraphDB SPARQL endpoint URL |
| `GRAPHDB_GRAPH` | For sync | — | Named graph IRI to query |
| `GRAPHDB_QUERY_FILE` | For sync | `get_relevant_software.rq` | Path to SPARQL query file |
| `GRAPHDB_USER` | No | — | GraphDB username (authenticated endpoints) |
| `GRAPHDB_PASSWORD` | No | — | GraphDB password |
| `OUTPUT_JSONLD` | No | `dataset/catalog.jsonld` | Path for raw JSON-LD snapshot |
| `OUTPUT_JSONL` | No | `dataset/catalog.jsonl` | Path for processed JSONL output |
| `IMAGE_META_CACHE_MAX` | No | `128` | Max entries in image metadata LRU cache |
| `GITHUB_TOKEN` | No | — | GitHub token for repository info tool |
| `LOGLEVEL_CONSOLE` | No | `WARNING` | Console log level |
| `LOGLEVEL_FILE` | No | `INFO` | File log level |
| `FILE_LOG` | No | `1` | Enable file logging |
| `LOG_DIR` | No | `logs` | Log file directory |
| `LOG_PROMPTS` | No | `0` | Save VLM prompts/images for debugging |
| `DEBUG` | No | `0` | Enable debug mode (sets file log level to DEBUG) |

---

## API Keys

### OPENAI_API_KEY

OpenAI API key — required when using a standard OpenAI endpoint in `config.yaml`.

```dotenv
OPENAI_API_KEY=sk-xxxx
```

**Where to get it**: [OpenAI API Keys](https://platform.openai.com/api-keys)

### EPFL_API_KEY

API key for the EPFL OpenAI-compatible inference endpoint. Required when `config.yaml` uses `api_key_env: "EPFL_API_KEY"` for the agent model.

```dotenv
EPFL_API_KEY=your-epfl-key
```

**Default endpoint**: `https://inference-rcp.epfl.ch/v1`

### EPFL_API_KEY_EMBEDDER

API key for the EPFL embedder and reranker endpoints. Used by the retrieval pipeline when `config.yaml` sets `api_key_env: "EPFL_API_KEY_EMBEDDER"`.

```dotenv
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key
```

!!! note
    If `EPFL_API_KEY_EMBEDDER` is not set, the reranker is **disabled** and raw FAISS similarity scores are used instead.

### GITHUB_TOKEN

GitHub personal access token for the `repo_info_batch` tool.

```dotenv
GITHUB_TOKEN=ghp_xxxx
```

**Permissions**: `public_repo` (read access)

**Required**: No — the tool falls back to DeepWiki MCP or repocards without it.

---

## Model Configuration

### CONFIG_PATH

Path to the YAML configuration file that defines agent model, available models, and retrieval settings.

```dotenv
CONFIG_PATH=config.yaml
```

**Default**: `config.yaml` (repository root)

See [Configuration Guide](../getting-started/configuration.md) for full `config.yaml` reference.

---

## Pipeline

### SOFTWARE_CATALOG

Path to the software catalog JSONL file used for startup embedding.

```dotenv
SOFTWARE_CATALOG=dataset/catalog.jsonl
```

**Default**: `dataset/catalog.jsonl`

### RAG_INDEX_DIR

Directory where FAISS index artifacts (`index.faiss`, `meta.json`) are stored and loaded.

```dotenv
RAG_INDEX_DIR=artifacts/rag_index
```

**Default**: `artifacts/rag_index`

### NUM_CHOICES

Default number of tool recommendations returned to the user. Overridable per-session from the UI.

```dotenv
NUM_CHOICES=3
```

**Default**: `3`

### EMBED_CATALOG_ON_START

Whether the pipeline embeds the full catalog at startup (useful if the FAISS index is empty or stale).

```dotenv
EMBED_CATALOG_ON_START=1   # 1=enabled, 0=disabled
```

**Default**: `1` (enabled)

Disable to speed up cold starts when you know the index is already populated.

### AGENT_OUTPUT_RETRIES

Number of times the agent retries structured-output validation before giving up.

```dotenv
AGENT_OUTPUT_RETRIES=3
```

**Default**: `3`

### AGENT_CACHE_MAX

Maximum number of agent instances kept in the LRU cache. Each unique `(model_name, base_url, api_key_env, num_choices)` combination gets its own cached instance.

```dotenv
AGENT_CACHE_MAX=16
```

**Default**: `16`

### IMAGE_META_CACHE_MAX

Maximum number of entries in the in-process image metadata LRU cache (keyed by file path + mtime + size).

```dotenv
IMAGE_META_CACHE_MAX=128
```

**Default**: `128`

---

## Catalog Sync

These variables control the catalog synchronization process (`ai_agent sync` and background refresh).

### GRAPHDB_URL

SPARQL endpoint URL of the GraphDB instance to query.

```dotenv
GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
```

**Required**: Yes, when running `ai_agent sync`.

### GRAPHDB_GRAPH

Named graph IRI to query. Must be an absolute IRI (starts with `http://` or `https://`).

```dotenv
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
```

**Required**: Yes, when running `ai_agent sync`.

### GRAPHDB_QUERY_FILE

Path to the SPARQL `.rq` query file. The query must contain a `{graph}` placeholder that is substituted with `GRAPHDB_GRAPH`.

```dotenv
GRAPHDB_QUERY_FILE=get_relevant_software.rq
```

**Default**: `get_relevant_software.rq`

### GRAPHDB_USER / GRAPHDB_PASSWORD

Credentials for authenticated GraphDB endpoints (optional).

```dotenv
GRAPHDB_USER=myuser
GRAPHDB_PASSWORD=mypassword
```

### OUTPUT_JSONLD

Path for the raw JSON-LD snapshot produced by the SPARQL fetch.

```dotenv
OUTPUT_JSONLD=dataset/catalog.jsonld
```

**Default**: `dataset/catalog.jsonld`

### OUTPUT_JSONL

Path for the processed JSONL catalog file.

```dotenv
OUTPUT_JSONL=dataset/catalog.jsonl
```

**Default**: `dataset/catalog.jsonl`

### SYNC_EVERY_HOURS

Interval (in hours) for background catalog refresh. Set to `0` to disable.

```dotenv
SYNC_EVERY_HOURS=24
```

**Default**: `0` (disabled)

**Minimum effective interval**: 60 seconds (clamped internally)

### SYNC_SKIP_IF_FRESH_SECONDS

Skip the remote SPARQL sync if the local catalog file is younger than this many seconds. Useful for fast container restarts.

```dotenv
SYNC_SKIP_IF_FRESH_SECONDS=3600
```

**Default**: `0` (disabled — always sync)

### SYNC_FORCE

Force a full sync regardless of the freshness check.

```dotenv
SYNC_FORCE=1
```

**Default**: `0`

---

## Logging

### LOGLEVEL_CONSOLE

Console log level.

```dotenv
LOGLEVEL_CONSOLE=WARNING
```

**Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`  
**Default**: `WARNING`

### LOGLEVEL_FILE

File log level.

```dotenv
LOGLEVEL_FILE=INFO
```

**Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`  
**Default**: `INFO` (or `DEBUG` when `DEBUG=1`)  
**Log files**: `logs/app_YYYYMMDD.log` (rotated at midnight, 14-day retention)

### FILE_LOG

Enable or disable file logging.

```dotenv
FILE_LOG=1
```

**Default**: `1` (enabled)

### LOG_DIR

Directory for log files. Created automatically if it does not exist.

```dotenv
LOG_DIR=logs
```

**Default**: `logs`

### LOG_PROMPTS

Save VLM prompt snapshots (text + images) to `logs/` for debugging.

```dotenv
LOG_PROMPTS=1
```

**Default**: `0`

!!! warning
    Enabling `LOG_PROMPTS` can consume significant disk space. Use only for short debugging sessions.

### DEBUG

Enable debug mode. When set, the file log level defaults to `DEBUG`.

```dotenv
DEBUG=1
```

**Default**: `0`

---

## Complete .env Example

```dotenv
# ── API Keys ────────────────────────────────────────────────────────────────
# Use either OPENAI_API_KEY (for OpenAI) or EPFL_API_KEY (for EPFL endpoint)
OPENAI_API_KEY=sk-xxxx
EPFL_API_KEY=your-epfl-key
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key
GITHUB_TOKEN=ghp_xxxx   # optional

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG_PATH=config.yaml

# ── Catalog ──────────────────────────────────────────────────────────────────
SOFTWARE_CATALOG=dataset/catalog.jsonl
RAG_INDEX_DIR=artifacts/rag_index

# ── Catalog Sync (required for ai_agent sync) ────────────────────────────────
GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
GRAPHDB_QUERY_FILE=get_relevant_software.rq
# GRAPHDB_USER=myuser
# GRAPHDB_PASSWORD=mypassword
SYNC_EVERY_HOURS=24
SYNC_SKIP_IF_FRESH_SECONDS=0
SYNC_FORCE=0

# ── Pipeline ──────────────────────────────────────────────────────────────────
NUM_CHOICES=3
EMBED_CATALOG_ON_START=1
AGENT_OUTPUT_RETRIES=3
AGENT_CACHE_MAX=16
IMAGE_META_CACHE_MAX=128

# ── Logging ───────────────────────────────────────────────────────────────────
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0
DEBUG=0
```

---

## Security Best Practices

!!! warning "Never commit `.env` files"
    Add `.env` to `.gitignore` to prevent accidental credential exposure.

!!! tip "Use `.env.example`"
    Provide a `.env.example` with dummy values so contributors know which variables to set.

## Next Steps

- [Configuration Guide](../getting-started/configuration.md)
- [CLI Commands](cli.md)
- [Architecture Overview](../architecture/overview.md)

