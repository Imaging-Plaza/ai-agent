# Changelog

All notable changes to the AI Imaging Agent are documented here.

For the complete, detailed changelog, see [CHANGELOG.md](https://github.com/imaging-plaza/ai-agent/blob/main/CHANGELOG.md) in the repository.

## Recent Releases

### [Unreleased]

#### Added
- **New chat-based interface** (`ai_agent chat`) with conversational AI assistant
- **Imaging Plaza branding**: Custom green theme and logo
- **Similarity-based query expansion**: Dynamic embedding-based matching
- **Iterative retrieval with retry**: Automatic retry when results insufficient
- **Agent alternative search tool**: Agent-driven search refinement
- **YAML model configuration**: Flexible multi-model support
- **DeepWiki MCP integration**: Fast repository info without rate limits

#### Changed
- CLI now supports `ai_agent chat` (removed `ai_agent ui`)
- Enhanced retrieval pipeline with automatic retry
- Simplified UI state management
- Removed legacy code paths

#### Removed
- `VLMToolSelector` class (replaced by PydanticAI agent)
- Legacy refine intent detection
- Outdated test files

### [0.1.3] - 2025-10-22

#### Added
- Gradio Space runner tool
- Repository info tool

#### Fixed
- Gradio UI context binding
- Chatbot message format migration
- Cache cleaning
- PNG preview handling

### [0.1.2] - 2025-10-07

#### Added
- Pydantic AI pipeline with tools
- Better runnable example handling

### [0.1.1] - 2025-10-02

#### Added
- Experimental Pydantic AI agent skeleton
- Multimodal agent pathway

### [0.1.0] - 2025-09-30

#### Added
- Initial chat functionality

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

## Contributing

All notable changes should be documented in [CHANGELOG.md](https://github.com/imaging-plaza/ai-agent/blob/main/CHANGELOG.md) following the [Keep a Changelog](https://keepachangelog.com/) format.

See [Contributing Guide](../development/contributing.md) for details.
