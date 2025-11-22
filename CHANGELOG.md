# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **YAML Model Configuration**: New `config.yaml` file for flexible model configuration supporting OpenAI, EPFL inference server, and any OpenAI-compatible API endpoints.
- **Multi-Model Support**: Can now configure different models for agent (main reasoning & tool selection) and image analysis independently.
- **Configuration Module**: New `utils/config.py` with Pydantic models for type-safe configuration loading and validation.

### Changed
- **Model Initialization**: Agent now uses configuration from `config.yaml`.
- **API Client Creation**: OpenAI clients now support custom `base_url` for alternative API endpoints (EPFL, custom deployments).
- **Dependency**: Added `pyyaml` to `pyproject.toml` dependencies.
- **.env.dist**: Updated with documentation about new config.yaml system and backward compatibility notes.
- **UI State Management Simplified**: Removed complex refine intent detection system. Agent now naturally handles requests for alternatives via conversation history without hard-coded heuristics.
- **UI Handler Simplified**: Reduced `handle_message()` parameters from 8 to 6, removing `last_task_state`, `last_suggestions_state`, and `excluded_names` state tracking.
- **Agent-Only Path**: Removed `USE_AGENT` conditional (always uses Pydantic AI agent). Deleted dead code path for non-agent pipeline invocation.

### Removed
- **VLMToolSelector**: Deleted unused `generator/generator.py` containing VLMToolSelector class. The pydantic-ai agent handles all tool selection directly.
- **Dead Functions**: Removed `is_refine_intent()` and `strip_refine_keywords()` from `utils/tags.py` along with `_REFINE_KEYWORDS` constant.
- **Legacy UI Code**: Removed `_load_catalog()` function (unused), complex refine intent detection logic (~60 lines), and base_task/prev_suggestions tracking.
- **Pipeline Simplification**: Removed `force_clarification` logic and `has_refine` import from `api/pipeline.py` (legacy code path never invoked by agent).
- **Legacy Method**: Removed `recommend_and_link()` method from `api/pipeline.py` (~180 lines) - only used by outdated tests, replaced by agent-based approach.
- **State Variables**: Removed 3 Gradio State objects: `last_task_state`, `last_suggestions_state`, `excluded_names`.
- **Outdated Tests**: Removed `tests/full_test.py` which only tested the removed `recommend_and_link()` method.

### Fixed
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