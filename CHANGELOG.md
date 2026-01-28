# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **New chat-based interface** (`ai_agent chat`) with conversational AI assistant
  - Chatbot component with rich media rendering (images, files, JSON, code blocks)
  - Inline file upload support for PNG, JPG, WEBP, TIFF, DICOM, NIfTI, CSV, JSON, XML, MP3, MP4
  - File previews with format-specific icons rendered in chat messages
  - Tool recommendation cards with detailed metadata (modality, dimensions, license, tags)
  - Demo execution as conversational flow - assistant asks "Would you like me to run the demo?"
  - Tool execution traces displayed as collapsible `<details>` sections after responses
  - Debug sidebar showing conversation state, excluded tools, and preview images
  - Full conversation context maintained across multi-turn interactions
  - Affirmative response detection for demo confirmations (yes, sure, ok, etc.)
- `respond(message, files, state) -> (reply, media, state)` core interface function
  - Encapsulates all agent logic in testable, UI-independent function
  - State management via `ChatState` dataclass with serialization
  - `ChatMessage` dataclass for rich reply composition with markdown, images, files, traces
- `handlers.py` module with agent response logic
- `components.py` module for reusable chat UI components
- `formatters.py` helpers for rich message and media formatting
- `state.py` chat state models and serialization utilities
- `visualizations.py` helpers for rendering previews, traces, and visual state
- `app.py` Gradio app implementing the chat UI
- **Imaging Plaza branding**: Custom CSS theme with Plaza green colors (#00A991)
- **Logo integration**: Official Imaging Plaza white logo displayed in header
- **Redesigned layout**: Reorganized UI with header banner, left chat panel, and right sidebar for files and state
- **Similarity-Based Query Expansion**: Replaced hard-coded synonym dictionaries with dynamic embedding-based similarity matching using BGE-M3 embeddings. Vocabulary is automatically extracted from catalog and updated on catalog changes.
- **Iterative Retrieval with Retry**: Added automatic retry logic (up to 2 attempts) when initial search returns insufficient results (<5 candidates). System generates alternative queries using semantic neighbors.
- **Agent Alternative Search Tool**: New `search_alternative` tool allows agent to explicitly request searches with different query formulations (up to 3 per conversation). Enables agent-driven iterative refinement.
- **YAML Model Configuration**: New `config.yaml` file for flexible model configuration supporting OpenAI, EPFL inference server, and any OpenAI-compatible API endpoints.
- **Multi-Model Support**: Can now configure different models for agent (main reasoning & tool selection).
- **Configuration Module**: New `utils/config.py` with Pydantic models for type-safe configuration loading and validation.

### Changed
- CLI now supports `ai_agent chat`
- **DeepWiki MCP integration**: Repository info tool now uses DeepWiki MCP server (https://mcp.deepwiki.com/sse) as primary source for GitHub repository documentation. DeepWiki provides fast, pre-indexed documentation access without API rate limits.
- Automatic fallback to `repocards` library (replacing previous direct GitHub API implementation) when DeepWiki is unavailable or times out, ensuring robust repository information retrieval for both indexed and newly-created repositories.
- Updated `pydantic-ai` dependency to include MCP support via `pydantic-ai[mcp]` extra.
- Enhanced `RepoSummaryOutput` schema to include `source` field indicating whether data came from "deepwiki" or "repocards".
- Repository info tool logs now track data source (DeepWiki vs repocards) for observability.
- Replaced previous direct GitHub API implementation with `repocards` library as the fallback mechanism for repository information retrieval.
- **YAML Model Configuration**: New `config.yaml` file for flexible model configuration supporting OpenAI, EPFL inference server, and any OpenAI-compatible API endpoints.
- **Multi-Model Support**: Can now configure different models for agent (main reasoning & tool selection).
- **Configuration Module**: New `utils/config.py` with Pydantic models for type-safe configuration loading and validation.
- **Query Expansion Method**: Moved from dictionary-based to similarity-based expansion using catalog vocabulary. Queries are now expanded with semantically related terms found via cosine similarity.
- **Retrieval Pipeline**: Enhanced `retrieve_no_rerank()` with automatic retry and alternative query generation when results are insufficient.
- **Agent Prompt**: Updated to explain new retrieval capabilities including similarity expansion, automatic retry, and when/how to use `search_alternative` tool.
- **Import Paths**: Fixed and standardized all import paths to use `ai_agent.` prefix for consistency.
- **Model Initialization**: Agent now uses configuration from `config.yaml`.
- **API Client Creation**: OpenAI clients now support custom `base_url` for alternative API endpoints (EPFL, custom deployments).
- **Dependency**: Added `pyyaml` to `pyproject.toml` dependencies.
- **.env.dist**: Updated with documentation about new config.yaml system and backward compatibility notes.
- **UI State Management Simplified**: Removed complex refine intent detection system. Agent now naturally handles requests for alternatives via conversation history without hard-coded heuristics.
- **UI Handler Simplified**: Reduced `handle_message()` parameters from 8 to 6, removing `last_task_state`, `last_suggestions_state`, and `excluded_names` state tracking.
- **Agent-Only Path**: Removed `USE_AGENT` conditional (always uses Pydantic AI agent). Deleted dead code path for non-agent pipeline invocation.
- **UI redesign**: File upload moved to dedicated right panel for cleaner workflow
- **Visual hierarchy**: Header with gradient green banner and logo
- **Button styling**: Primary actions use Imaging Plaza green theme colors

### Removed
- **VLMToolSelector**: Deleted unused `generator/generator.py` containing VLMToolSelector class. The pydantic-ai agent handles all tool selection directly.
- **Dead Functions**: Removed `is_refine_intent()` and `strip_refine_keywords()` from `utils/tags.py` along with `_REFINE_KEYWORDS` constant.
- **Legacy UI Code**: Removed `_load_catalog()` function (unused), complex refine intent detection logic (~60 lines), and base_task/prev_suggestions tracking.
- **Pipeline Simplification**: Removed `force_clarification` logic and `has_refine` import from `api/pipeline.py` (legacy code path never invoked by agent).
- **Legacy Method**: Removed `recommend_and_link()` method from `api/pipeline.py` (~180 lines) - only used by outdated tests, replaced by agent-based approach.
- **State Variables**: Removed 3 Gradio State objects: `last_task_state`, `last_suggestions_state`, `excluded_names`.
- **Outdated Tests**: Removed `tests/full_test.py` which only tested the removed `recommend_and_link()` method.
- CLI no more supports `ai_agent ui` command

### Fixed
- **Pydantic Forward Reference**: Reordered class definitions in `schema.py` so `Conversation` and `ConversationStatus` are defined before `ToolSelection` to prevent "class-not-fully-defined" errors.
- **Conversation Context**: Agent now properly maintains conversation history, enabling natural understanding of follow-up requests like "show me alternatives".
- **Clear Button**: Disabled during processing to prevent race conditions with ongoing requests.
- **Alternative Tool Requests**: All recommended tools are now automatically added to the exclusion list (banlist) and properly passed to the agent through AgentState, ensuring follow-up requests like "I would like another tool" correctly return different tools.
- **History Table**: Follow-up requests (without files) no longer create duplicate history entries. Only primary requests with files are logged to the History table.

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