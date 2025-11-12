# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **DeepWiki MCP integration**: Repository info tool now uses DeepWiki MCP server (https://mcp.deepwiki.com/sse) as primary source for GitHub repository documentation. DeepWiki provides fast, pre-indexed documentation access without API rate limits.
- Automatic fallback to GitHub API when DeepWiki is unavailable or times out, ensuring robust repository information retrieval for both indexed and newly-created repositories.

### Changed
- Updated `pydantic-ai` dependency to include MCP support via `pydantic-ai[mcp]` extra.
- Enhanced `RepoSummaryOutput` schema to include `source` field indicating whether data came from "deepwiki" or "github_api".
- Repository info tool logs now track data source (DeepWiki vs GitHub API) for observability.

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