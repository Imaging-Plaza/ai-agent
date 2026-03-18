# Project Guide

This guide is a practical map of the entire repository for contributors and maintainers.

It focuses on:
- What each folder is responsible for
- Which Python environment and package workflow are the defaults
- Which commands are currently valid
- What to improve next in architecture, testing, performance, and developer experience

## 1) System Summary

AI Imaging Agent is a RAG plus VLM recommender for imaging software.

High-level flow:
1. User uploads file(s) and enters a task.
2. Retrieval stage finds candidate tools (BGE-M3 + FAISS + reranker).
3. Agent/VLM stage ranks candidates with image-aware reasoning.
4. UI renders ranked recommendations and optional demo links.

Primary orchestrator: [src/ai_agent/api/pipeline.py](src/ai_agent/api/pipeline.py)

## 2) Default Python Environment And Packages (Dev Container Canonical)

Assume development is done inside the dev container.

Source of truth:
- Dev container: [.devcontainer/devcontainer.json](../.devcontainer/devcontainer.json)
- Package metadata and pinned dependencies: [pyproject.toml](../pyproject.toml)
- Secondary dependency list: [requirements.txt](../requirements.txt)

Default environment:
- OS: Debian Bookworm (dev container)
- Python: 3.12
- Environment manager: uv
- Virtual environment path: .venv

Recommended commands:

```bash
uv venv
uv pip install -e .
uv pip install -e ".[dev]"
```

Run and test:

```bash
ai_agent chat
ai_agent sync
pytest tests/
```

Important note on command drift:
- CLI officially supports `chat` and `sync` in [src/ai_agent/cli.py](../src/ai_agent/cli.py).
- [justfile](../justfile) currently references `ai_agent ui`, which does not match current CLI modes.
- Documentation in this guide follows the actual CLI implementation.

## 3) Repository Top-Level Map

- [.github/](../.github/): automation and agent instructions
- [.devcontainer/](../.devcontainer/): dev container build and editor defaults
- [docs/](.): MkDocs source pages
- [src/](../src/): application source code
- [tests/](../tests/): test suite
- [data/](../data/): sample data assets
- [tools/](../tools/): container/tooling helpers
- [CHANGELOG.md](../CHANGELOG.md): release history
- [config.yaml](../config.yaml): model/provider configuration
- [mkdocs.yml](../mkdocs.yml): docs site navigation and theme
- [pyproject.toml](../pyproject.toml): package metadata, dependencies, entrypoints

## 4) Detailed Source Folder Responsibilities

Package root: [src/ai_agent/](../src/ai_agent)

### 4.1 [src/ai_agent/agent/](../src/ai_agent/agent)

Purpose: conversational orchestration using PydanticAI.

Key files:
- [src/ai_agent/agent/agent.py](../src/ai_agent/agent/agent.py): agent setup, tool wiring, response flow
- [src/ai_agent/agent/models.py](../src/ai_agent/agent/models.py): state/output models
- [src/ai_agent/agent/utils.py](../src/ai_agent/agent/utils.py): helper utilities and guardrails
- [src/ai_agent/agent/tools/](../src/ai_agent/agent/tools): concrete tool implementations
- [src/ai_agent/agent/tools/mcp/](../src/ai_agent/agent/tools/mcp): MCP adapters

Boundary:
- Should orchestrate tools and policy, not own retrieval internals.

### 4.2 [src/ai_agent/api/](../src/ai_agent/api)

Purpose: pipeline orchestration between inputs, retrieval, and selection.

Key file:
- [src/ai_agent/api/pipeline.py](../src/ai_agent/api/pipeline.py)

Responsibilities:
- validate files
- extract metadata
- build retrieval query
- call retrieval and selection stages
- manage index refresh/reload behavior

Boundary:
- Keep UI concerns out of this module.

### 4.3 [src/ai_agent/retriever/](../src/ai_agent/retriever)

Purpose: deterministic retrieval stack (no LLM calls).

Key files:
- [src/ai_agent/retriever/text_embedder.py](../src/ai_agent/retriever/text_embedder.py)
- [src/ai_agent/retriever/vector_index.py](../src/ai_agent/retriever/vector_index.py)
- [src/ai_agent/retriever/reranker.py](../src/ai_agent/retriever/reranker.py)
- [src/ai_agent/retriever/software_doc.py](../src/ai_agent/retriever/software_doc.py)

Boundary:
- Retrieval quality logic should stay here.

### 4.4 [src/ai_agent/generator/](../src/ai_agent/generator)

Purpose: selection schema and prompting primitives.

Key files:
- [src/ai_agent/generator/prompts.py](../src/ai_agent/generator/prompts.py)
- [src/ai_agent/generator/schema.py](../src/ai_agent/generator/schema.py)

Boundary:
- Keep this layer focused on schema and prompt contracts, not transport/UI concerns.

### 4.5 [src/ai_agent/ui/](../src/ai_agent/ui)

Purpose: Gradio app and interaction handling.

Key files:
- [src/ai_agent/ui/app.py](../src/ai_agent/ui/app.py)
- [src/ai_agent/ui/handlers.py](../src/ai_agent/ui/handlers.py)
- [src/ai_agent/ui/components.py](../src/ai_agent/ui/components.py)
- [src/ai_agent/ui/formatters.py](../src/ai_agent/ui/formatters.py)
- [src/ai_agent/ui/state.py](../src/ai_agent/ui/state.py)
- [src/ai_agent/ui/visualizations.py](../src/ai_agent/ui/visualizations.py)

Boundary:
- UI should call orchestrators, not reimplement retrieval/selection decisions.

### 4.6 [src/ai_agent/utils/](../src/ai_agent/utils)

Purpose: cross-cutting utility functions.

Key files:
- [src/ai_agent/utils/config.py](../src/ai_agent/utils/config.py)
- [src/ai_agent/utils/file_validator.py](../src/ai_agent/utils/file_validator.py)
- [src/ai_agent/utils/image_meta.py](../src/ai_agent/utils/image_meta.py)
- [src/ai_agent/utils/image_io.py](../src/ai_agent/utils/image_io.py)
- [src/ai_agent/utils/previews.py](../src/ai_agent/utils/previews.py)
- [src/ai_agent/utils/tags.py](../src/ai_agent/utils/tags.py)
- [src/ai_agent/utils/temp_file_manager.py](../src/ai_agent/utils/temp_file_manager.py)

Boundary:
- Keep utilities reusable and independent from UI-specific logic.

### 4.7 [src/ai_agent/catalog/](../src/ai_agent/catalog)

Purpose: catalog synchronization and refresh helpers.

Key file:
- [src/ai_agent/catalog/sync.py](../src/ai_agent/catalog/sync.py)

Boundary:
- Catalog IO and sync logic should stay isolated from ranking logic.

### 4.8 [src/ai_agent/core/](../src/ai_agent/core)

Purpose: shared core coordination such as pipeline registry.

Key file:
- [src/ai_agent/core/pipeline_registry.py](../src/ai_agent/core/pipeline_registry.py)

Boundary:
- Keep core primitives minimal and dependency-light.

### 4.9 [src/ai_agent/queries/](../src/ai_agent/queries)

Purpose: query assets used by catalog sync/retrieval support.

Key file:
- [src/ai_agent/queries/get_relevant_software.rq](../src/ai_agent/queries/get_relevant_software.rq)

Boundary:
- Keep query definitions versioned and testable.

### 4.10 [src/ai_agent/cli.py](../src/ai_agent/cli.py)

Purpose: command entry point and mode dispatch.

Current modes:
- `chat`
- `sync`

This is the command contract docs should follow.

## 5) Supporting Folders

### 5.1 [tests/](../tests)

Contains unit/integration tests and test fixtures under [tests/data/](../tests/data).

Improvement target:
- add more focused tests for UI handler edge cases and tool failure handling.

### 5.2 [tools/](../tools)

Container and deployment support assets.

Notable file:
- [tools/image/Dockerfile](../tools/image/Dockerfile) (uv + Python 3.12 baseline)

### 5.3 [docs/](.)

Documentation source for MkDocs.

Add new pages to [mkdocs.yml](../mkdocs.yml) nav to keep docs discoverable.

## 6) Known Inconsistencies To Track

1. [justfile](../justfile) uses `ai_agent ui`, while [src/ai_agent/cli.py](../src/ai_agent/cli.py) defines `chat` and `sync`.
2. Installation docs often show pip-first flow, while dev container bootstrap is uv-first.
3. [requirements.txt](../requirements.txt) is looser than [pyproject.toml](../pyproject.toml), which contains current pinned/runtime dependencies.

## 7) Codebase Improvement Guidelines

### 7.1 Architecture And Modularity

1. Keep strict stage boundaries: retrieval logic in `retriever`, selection contracts in `generator`, orchestration in `api`.
2. Minimize cross-layer imports from `ui` to low-level modules.
3. Introduce lightweight interface contracts for tool adapters to reduce coupling in `agent/tools`.
4. Centralize shared constants/env defaults to reduce duplicated configuration behavior.

### 7.2 Testing And Quality Gates

1. Add regression tests for format-token query construction and retry broadening behavior.
2. Add failure-path tests for image preview generation and graceful degradation.
3. Add contract tests for agent tool outputs (search, alternative search, repo info).
4. Enforce formatting/lint/type checks in CI (`ruff`, `black --check`, `mypy`, `pytest`).

### 7.3 Performance And Retrieval Quality

1. Add benchmark fixtures for retrieval latency and reranker throughput.
2. Track retrieval quality with a small fixed evaluation set (top-k recall, MRR).
3. Cache expensive metadata extraction where safe for repeated files in a session.
4. Make index reload behavior observable with structured counters in logs.

### 7.4 Developer Experience And CI

1. Align `just` tasks with real CLI contract (`chat`/`sync`).
2. Add a docs link checker in CI to prevent markdown drift.
3. Document one canonical local workflow (dev container first, optional local pip fallback).
4. Add a short maintainer checklist for release prep and changelog updates.

## 8) Practical Contributor Checklist

Before opening a PR:
1. Install/update in editable mode in the active environment.
2. Run tests relevant to changed modules.
3. Validate docs links if docs were touched.
4. Update [CHANGELOG.md](../CHANGELOG.md) for user-visible changes.
5. Confirm command and environment docs still match real behavior.

## 9) Related References

- [README.md](../README.md)
- [docs/index.md](index.md)
- [docs/architecture/overview.md](architecture/overview.md)
- [docs/development/structure.md](development/structure.md)
- [AGENTS.md](../AGENTS.md)
- [.github/copilot-instructions.md](../.github/copilot-instructions.md)
