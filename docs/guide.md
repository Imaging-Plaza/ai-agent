# Project Guide

This guide is a practical repository map for contributors and maintainers.

It focuses on:

- What each folder owns
- Which environment and commands are canonical
- How the React frontend, FastAPI backend, agent, and retrieval pipeline fit together
- Where to improve the project next

## 1. System Summary

AI Imaging Agent is a RAG plus VLM recommender for imaging software.

High-level flow:

1. User signs in to the React app when `APP_PASSWORD` is set.
2. User uploads file(s), reuses session assets, or asks a text-only question.
3. FastAPI stores assets, builds previews/views, and streams chat events over SSE.
4. The PydanticAI agent searches the catalog, asks for alternatives or repo info when needed, and returns recommendations.
5. Retrieval logic builds metadata-aware queries, searches FAISS, reranks candidates, and feeds ranked candidates back to the agent.
6. The React UI renders recommendations, media, traces, clarification prompts, and pending demo actions.

Primary runtime entry point for the new frontend: `ai_agent serve`.

Legacy UI entry point: `ai_agent chat`.

## 2. Default Environment

Assume development is done inside the dev container unless a task says otherwise.

Source of truth:

- Dev container: `.devcontainer/devcontainer.json`
- Python package metadata: `pyproject.toml`
- Frontend package metadata: `src/frontend/package.json`
- Model/retrieval config: `config.yaml`

Default environment:

- OS: Debian Bookworm in the dev container
- Python: 3.12 in the dev container
- Package manager: `uv`
- Virtual environment path: `.venv`
- Frontend runtime: Node.js 20+

Recommended Python setup:

```bash
uv venv
uv pip install -e .
uv pip install -e ".[dev]"
```

Recommended frontend setup:

```bash
cd src/frontend
npm install
```

Run and test:

```bash
ai_agent serve
ai_agent chat
ai_agent sync
pytest tests/
```

Frontend checks:

```bash
cd src/frontend
npm run lint
npm run build
```

## 3. Repository Top-Level Map

- `.github/`: automation and agent instructions
- `.devcontainer/`: dev container build and editor defaults
- `docs/`: MkDocs source pages
- `src/ai_agent/`: Python backend, agent, retrieval, services, and legacy Gradio UI
- `src/frontend/`: React + Vite frontend
- `tests/`: Python test suite
- `data/`: sample data assets
- `tools/`: container/tooling helpers
- `CHANGELOG.md`: release history
- `config.yaml`: model/provider/retrieval configuration
- `Dockerfile`: production image that builds frontend and runs FastAPI
- `docker-compose.yml`: container plus Cloudflare tunnel sidecar
- `mkdocs.yml`: docs site navigation and theme
- `pyproject.toml`: Python package metadata, dependencies, entrypoints

## 4. Source Responsibilities

### `src/frontend/`

Purpose: React SPA for the primary user experience.

Key areas:

- `src/frontend/src/pages/ChatPage.tsx`: main chat shell
- `src/frontend/src/pages/LoginPage.tsx`: passphrase login
- `src/frontend/src/components/`: chat input, message list, sidebar, model picker, asset modals, volume renderer, recommendation cards
- `src/frontend/src/hooks/`: auth, chat, conversations, transcription, theme state
- `src/frontend/src/lib/`: API wrapper, SSE client, slash commands, backup/date utilities
- `src/frontend/src/workers/`: browser worker support
- `src/frontend/public/examples/`: example prompt assets

Boundary:

- Frontend calls `/api/*`; it should not duplicate recommendation logic.
- Local browser storage can keep transcripts, but server assets are session-scoped and may disappear after backend restart.

### `src/ai_agent/api/`

Purpose: FastAPI app, API routers, dependency wiring, schemas, and pipeline access.

Key files:

- `server.py`: app factory, CORS, router mounting, production SPA serving
- `schemas.py`: request/response models
- `deps.py`: auth and shared dependency helpers
- `routers/auth.py`: passphrase login/logout/status
- `routers/chat.py`: SSE chat, approvals, declines, demo confirmation
- `routers/files.py`: upload, previews, raw files, volume bytes, slice/MIP views
- `routers/models.py`: model picker options from `config.yaml`
- `routers/catalog.py`, `routers/health.py`: catalog and health endpoints
- `pipeline.py`: RAG retrieval orchestration

Boundary:

- API/service modules own transport and session concerns.
- Retrieval internals stay in `retriever/`.
- Agent behavior stays in `agent/`.

### `src/ai_agent/services/`

Purpose: stateful service layer behind FastAPI.

Key files:

- `sessions.py`: in-memory sessions, assets, and pending actions
- `files.py`: file ingestion and preview/metadata registration
- `chat.py`: synchronous chat turn processing and pending action handling
- `views.py`: volume info, slices, MIPs, and raw volume bytes

Boundary:

- Services bridge API routes and core agent/pipeline behavior.
- Keep UI rendering decisions in the frontend.

### `src/ai_agent/agent/`

Purpose: conversational orchestration using PydanticAI.

Key files:

- `agent.py`: agent setup, tool wiring, response flow
- `models.py`: state/output models
- `utils.py`: helper utilities and guardrails
- `tools/`: search, alternatives, repo info, demo discovery, MCP adapters

Boundary:

- Agent orchestrates tools and policy; retrieval quality logic stays in `retriever/`.

### `src/ai_agent/retriever/`

Purpose: deterministic retrieval stack.

Key files:

- `text_embedder.py`: remote/local embedding
- `vector_index.py`: FAISS index management
- `reranker.py`: remote/local reranking
- `software_doc.py`: catalog schema and loading
- `utils.py`: retrieval helpers

Boundary:

- No UI code and no conversation transport.

### `src/ai_agent/generator/`

Purpose: selection schemas and prompting primitives.

Key files:

- `prompts.py`
- `schema.py`

Boundary:

- Keep schema and prompt contracts here, not API or UI behavior.

### `src/ai_agent/ui/`

Purpose: legacy Gradio interface.

This remains available through `ai_agent chat`, but the React/FastAPI stack is the primary UI path.

### `src/ai_agent/utils/`

Purpose: cross-cutting utilities.

Key files:

- `config.py`
- `file_validator.py`
- `image_meta.py`
- `image_io.py`
- `previews.py`
- `tags.py`
- `temp_file_manager.py`

Boundary:

- Keep utilities reusable and independent from UI-specific logic.

### `src/ai_agent/catalog/`

Purpose: catalog synchronization and refresh helpers.

Key file:

- `sync.py`

## 5. Command Contract

Current CLI modes in `src/ai_agent/cli.py`:

- `ai_agent serve`: FastAPI backend for React; serves built SPA when available
- `ai_agent chat`: legacy Gradio UI
- `ai_agent sync`: one-shot catalog refresh

Documentation and scripts should follow this contract.

## 6. Known Inconsistencies To Track

1. Some older comments and docs may still call `chat` the main UI path.
2. `pyproject.toml` description still mentions Gradio, while the primary UI is React/FastAPI.
3. `requirements.txt` is looser than `pyproject.toml`, which is the stronger dependency source.
4. Session storage is in-memory; frontend transcript restore can outlive backend assets.

## 7. Improvement Guidelines

### Architecture

1. Keep frontend transport code in `src/frontend/src/lib`.
2. Keep API route handlers thin; move reusable behavior into `services/`.
3. Keep retrieval in `retriever/`, schema/prompt contracts in `generator/`, and orchestration in `agent/` or `api/`.
4. Avoid adding business logic to React components beyond presentation and client-side interaction state.

### Testing

1. Add API route tests for auth, files, chat approval/decline, and model listing.
2. Add frontend build/type checks to CI.
3. Add regression tests for volume view endpoints.
4. Keep retrieval quality checks small and reproducible.

### Developer Experience

1. Align task runners with `serve`, `chat`, and `sync`.
2. Add docs link validation in CI.
3. Document one canonical local development workflow.
4. Add a release checklist that includes Python tests, frontend build, docs build, and changelog updates.

## 8. Contributor Checklist

Before opening a PR:

1. Confirm behavior from executable code.
2. Run relevant Python tests.
3. Run `npm run lint` and `npm run build` for frontend changes.
4. Update user docs for user-facing changes.
5. Update `CHANGELOG.md` for user-visible changes.
6. Verify README, `docs/index.md`, and this guide still agree.

## 9. Related References

- `README.md`
- [docs/index.md](index.md)
- [Architecture Overview](architecture/overview.md)
- [Project Structure](development/structure.md)
- [CLI Reference](reference/cli.md)
- `AGENTS.md`
- `.github/copilot-instructions.md`
