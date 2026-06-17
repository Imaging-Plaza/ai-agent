# Project Structure

AI Imaging Agent is organized around a React frontend, FastAPI backend, PydanticAI agent, and retrieval pipeline.

## Directory Layout

```text
ai-agent/
├── .github/
│   └── copilot-instructions.md
├── artifacts/
│   └── rag_index/
├── data/
│   └── sample.jsonl
├── docs/
├── logs/
├── src/
│   ├── ai_agent/
│   │   ├── agent/
│   │   ├── api/
│   │   ├── catalog/
│   │   ├── core/
│   │   ├── generator/
│   │   ├── queries/
│   │   ├── retriever/
│   │   ├── services/
│   │   ├── ui/
│   │   └── utils/
│   └── frontend/
│       ├── public/
│       └── src/
├── tests/
├── config.yaml
├── docker-compose.yml
├── Dockerfile
├── mkdocs.yml
├── pyproject.toml
└── README.md
```

## Frontend

`src/frontend/` contains the React + Vite SPA.

```text
src/frontend/
├── index.html
├── package.json
├── vite.config.ts
├── public/
│   └── examples/
└── src/
    ├── App.tsx
    ├── main.tsx
    ├── index.css
    ├── components/
    ├── hooks/
    ├── lib/
    ├── pages/
    └── workers/
```

Important files:

- `pages/ChatPage.tsx`: main chat screen
- `pages/LoginPage.tsx`: password screen
- `components/ChatInput.tsx`: composer, uploads, gallery hooks, slash commands
- `components/MessageList.tsx`: conversation rendering
- `components/RecommendationCard.tsx`: recommendation display
- `components/Volume3D.tsx`: Three.js volume rendering
- `components/ModelPicker.tsx`: model/top-k/choice controls
- `hooks/useChat.tsx`: chat state and SSE events
- `hooks/useConversations.tsx`: local transcript persistence
- `lib/api.ts`: typed HTTP API wrapper
- `lib/sse.ts`: SSE helpers
- `lib/slashCommands.ts`: slash command parsing

## Python Package

`src/ai_agent/` contains the backend, agent, retrieval, and legacy UI.

### `api/`

FastAPI application and route layer.

```text
api/
├── server.py
├── schemas.py
├── deps.py
├── pipeline.py
└── routers/
    ├── auth.py
    ├── catalog.py
    ├── chat.py
    ├── files.py
    ├── health.py
    └── models.py
```

Responsibilities:

- Serve `/api/*` routes.
- Authenticate with `APP_PASSWORD` when configured.
- Stream chat events over SSE.
- Serve uploads, previews, raw assets, slices, MIPs, and volumes.
- Serve the built React bundle when `FRONTEND_DIST_DIR` exists.

### `services/`

Stateful service layer behind the API.

```text
services/
├── chat.py
├── files.py
├── sessions.py
└── views.py
```

Responsibilities:

- In-memory sessions and assets.
- File ingestion and metadata/preview registration.
- Chat turn processing and pending actions.
- Volume info/slice/MIP/byte extraction.

### `agent/`

PydanticAI conversational agent.

```text
agent/
├── agent.py
├── models.py
├── utils.py
└── tools/
    ├── deepwiki_tool.py
    ├── gradio_space_tool.py
    ├── query_utils.py
    ├── repo_info_tool.py
    ├── search_alternative_tool.py
    ├── search_tool.py
    └── mcp/
```

Responsibilities:

- Tool orchestration.
- Recommendation assembly.
- Alternative searches.
- Repository and demo context.
- MCP-backed tools.

### `retriever/`

Deterministic text retrieval stack.

```text
retriever/
├── text_embedder.py
├── vector_index.py
├── reranker.py
├── software_doc.py
└── utils.py
```

Responsibilities:

- Load catalog documents.
- Build/load FAISS index artifacts.
- Embed metadata-aware queries.
- Rerank candidate tools.

### `generator/`

Prompt and schema contracts.

```text
generator/
├── prompts.py
└── schema.py
```

### `utils/`

Reusable helpers.

```text
utils/
├── cache_db.py
├── config.py
├── file_validator.py
├── image_io.py
├── image_meta.py
├── previews.py
├── shutdown.py
├── tags.py
└── temp_file_manager.py
```

### `ui/`

Legacy Gradio UI launched by:

```bash
ai_agent chat
```

Keep it working, but use `src/frontend/` plus `ai_agent serve` for new frontend work.

### `catalog/`, `core/`, `queries/`

- `catalog/sync.py`: GraphDB sync and JSON-LD to JSONL conversion.
- `core/pipeline_registry.py`: shared pipeline singleton.
- `queries/get_relevant_software.rq`: SPARQL query asset.

### `cli.py`

Current command modes:

- `ai_agent serve`: FastAPI backend for React
- `ai_agent chat`: legacy Gradio UI
- `ai_agent sync`: catalog sync

## Tests

`tests/` contains Python tests for retrieval, pipeline behavior, catalog loading, repository info, query utilities, and preview caching.

Recommended command:

```bash
pytest tests/
```

Frontend checks live in `src/frontend`:

```bash
npm run lint
npm run build
```

## Configuration Files

### `pyproject.toml`

Python metadata, dependencies, package discovery, and console scripts.

### `src/frontend/package.json`

Frontend dependencies and scripts:

- `npm run dev`
- `npm run build`
- `npm run preview`
- `npm run lint`

### `config.yaml`

Agent model, model picker entries, embedder, and reranker configuration.

### `.env`

Runtime secrets and overrides such as API keys, `APP_PASSWORD`, `PORT`, and catalog paths.

### `Dockerfile`

Multi-stage build:

1. Node builds `src/frontend`.
2. Python installs the package.
3. Container runs `ai_agent serve`.

## Module Boundaries

- `src/frontend/`: browser UI and client-side state only.
- `api/`: HTTP/SSE transport and dependency wiring.
- `services/`: stateful backend use cases.
- `agent/`: conversational decisions and tool orchestration.
- `retriever/`: retrieval quality and indexing.
- `generator/`: prompt/schema contracts.
- `utils/`: reusable utilities with minimal dependencies.
- `catalog/`: catalog sync.

## Import Patterns

Python imports should use absolute package paths:

```python
from ai_agent.retriever.vector_index import VectorIndex
from ai_agent.utils.config import load_config
```

Frontend imports should keep shared API/SSE logic in `src/frontend/src/lib` and avoid scattering raw endpoint strings through components.

## Extension Points

### Add A Frontend View

Create or update components in `src/frontend/src/components`, add page-level state in `pages`, and centralize backend calls in `lib`.

### Add An API Route

Add a router in `src/ai_agent/api/routers`, schemas in `api/schemas.py`, and reusable behavior in `services/`.

### Add A Tool

Add implementation in `src/ai_agent/agent/tools` and register it with the agent.

### Add Retrieval Behavior

Keep embedding, FAISS, reranking, and catalog document behavior inside `retriever/`.

## Next Steps

- Learn about [Contributing](contributing.md)
- Explore [Testing](testing.md)
- Return to [Architecture Overview](../architecture/overview.md)
