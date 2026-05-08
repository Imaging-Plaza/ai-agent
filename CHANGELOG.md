# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- **Documentation sync**: Updated all documentation files to match current implementation exactly
  - `docs/architecture/retrieval.md`: Fixed embedder model (Qwen3-Embedding-8B via remote EPFL endpoint, not local BGE-M3); fixed reranker (BAAI/bge-reranker-v2-m3 remote)
  - `docs/architecture/agent.md`: Fixed default model (`openai/gpt-oss-120b` EPFL, not `gpt-4o-mini`); fixed tool names (`repo_info_batch`, not `repo_info`); updated tool caps and agent setup code
  - `docs/architecture/catalog.md`: Added full GraphDB SPARQL sync flow, mermaid diagram, all required environment variables
  - `docs/reference/environment.md`: Complete rewrite — comprehensive table of 27+ env vars including EPFL keys, GRAPHDB_* vars, SYNC_* vars, EMBED_CATALOG_ON_START, AGENT_CACHE_MAX, AGENT_OUTPUT_RETRIES, IMAGE_META_CACHE_MAX, DEBUG
  - `docs/getting-started/configuration.md`: Fixed default model and config.yaml example to match actual defaults; added EPFL API key docs and GraphDB sync vars; added local retrieval instructions
  - `docs/development/structure.md`: Added `core/` module, `queries/` directory, updated `agent/tools/` listing with all tool files and mcp/ subdir; fixed generator schema models; improved retriever pipeline description
  - `docs/reference/cli.md`: Updated `ai_agent sync` to describe actual SPARQL/GraphDB mechanism and required env vars
  - `docs/index.md`: Updated retrieval stack description to Qwen3-Embedding-8B + BGE-M3

## [1.0.0]

### 🚀 Major Features

- **Chat-based AI interface** (`ai_agent chat`)
  - Conversational assistant with rich media support (images, files, JSON, code)
  - Inline file uploads (PNG, JPG, TIFF, DICOM, NIfTI)
  - Tool recommendation cards and execution traces
  - Debug sidebar with conversation state and previews
  - Persistent multi-turn conversation context

- **Tool execution workflow**
  - Button-based approval system for safe tool execution
  - Inline “Tool Recommendation” UI with contextual actions
  - Downloadable results section for tool outputs
  - Separation of preview vs original result files

- **Tooling system (MCP integration)**
  - Centralized tool registry with dynamic lookup
  - Catalog-to-tool name mapping
  - Lazy loading for heavy tools
  - First integrated tool: 3D lung segmentation (HuggingFace Space)

---

### 🧠 Retrieval & Agent Improvements

- Iterative retrieval with retry for low-result queries
- Alternative search tool (`search_alternative`) for agent-driven refinement
- Batch repository verification via `repo_info_batch`
- Improved agent prompt and structured output reliability

---

### ⚙️ Configuration & Model Support

- YAML-based configuration (`config.yaml`)
  - Supports OpenAI, EPFL, and OpenAI-compatible APIs
- Configurable:
  - Agent model
  - Embedding backend
  - Reranker backend
- Type-safe config loading via Pydantic

---

### 🎨 UI & UX

- Redesigned layout with:
  - Header banner and Imaging Plaza branding
  - Chat panel + sidebar separation
- Tool analytics:
  - Execution frequency charts
  - Timeline visualization
- Improved file handling and previews

---

### ⚡ Performance Improvements

- Startup catalog pre-embedding (FAISS warm start)
- Metadata caching for images
- Preview generation caching
- Agent instance caching for custom models
- Repository info caching with TTL + deduplication
- Port pre-selection to avoid UI startup retries

---

### 🏗️ Architecture

- Modular structure:
  - handlers.py, components.py, state.py, formatters.py, visualizations.py
- Core interface:
  - `respond(message, files, state)`
- Clear separation of UI, agent logic, and tool execution
- MCP tools moved to dedicated subpackage

---

### 🧹 Cleanup & Simplification

- Removed:
  - Legacy non-agent pipeline
  - Fast mode
  - Text-based tool approval
  - Refine intent system
  - Deprecated CLI (`ai_agent ui`)
- Simplified UI state management
- Standardized import paths (`ai_agent.*`)

---

### 🛠️ Fixes

- Fixed startup refresh issues and duplicate sync
- Improved structured output validation reliability
- Prevented retrieval query drift
- Fixed FAISS rebuild when embeddings change
- Resolved Gradio compatibility issues
- Fixed race conditions and duplicate history entries
- Ensured consistent UI event outputs

## [0.1.3] - 2025-10-22

### Added
- Gradio space runner tool working on the drag-and-dropped image.
- Repo info tool extracting relevant information in markdown for the agent.

### Fixed
- Gradio UI: bound `clear.click(...)` inside the Blocks context to prevent “Cannot call click outside of a gradio.Blocks context”.
- Chatbot: migrated to `type="messages"` by adding a conversion shim (pairs <-> messages) in the handler to satisfy the new schema.
- UI polish: hide the “Demo link” textbox and “Run demo on preview” button at launch and until a tool is found; reveal them after recommendations appear and on selection, and hide again on Clear.
- Cache is now cleaned correctly.
- Fixed the preview for the png files.

## [0.1.2] - 2025-10-07

### Added
- Pydantic AI pipeline working with a few tools.
- Better handling of the runnable example link and reranker.

## [0.1.1] - 2025-10-02

### Added
- Experimental Pydantic AI agent skeleton with tool-based retrieval (feature flag `USE_AGENT=1`).
- Multimodal agent pathway (image preview + candidates) producing structured `ToolSelection` output.

## [0.1.0] - 2025-09-30

### Added
- Chat functionality.