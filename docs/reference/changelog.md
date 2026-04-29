# Changelog

All notable changes to the AI Imaging Agent are documented here.

For the complete, detailed changelog, see [CHANGELOG.md](https://github.com/imaging-plaza/ai-agent/blob/main/CHANGELOG.md) in the repository.

## Recent Releases

### [1.0.0]

#### 🚀 Added

- **Chat-based interface** (`ai_agent chat`) with conversational AI assistant and tool integration  
- **Imaging Plaza UI**: Custom branding, theme, and improved layout  
- **Iterative retrieval with automatic retry** for low-result queries  
- **Alternative search tool** for agent-driven query refinement  
- **YAML configuration** (`config.yaml`) for flexible model and backend setup  
- **DeepWiki MCP integration** for fast GitHub repository documentation access  

---

#### 🔄 Changed

- CLI updated: `ai_agent chat` replaces deprecated `ai_agent ui`  
- Retrieval pipeline enhanced with smarter expansion and retry logic  
- UI state management simplified  
- Agent-based architecture fully replaces legacy pipelines  

---

#### 🧹 Removed

- `VLMToolSelector` (replaced by agent-based tool selection)  
- Legacy refine intent detection system  
- Deprecated UI command (`ai_agent ui`)  
- Outdated tests and unused code paths  

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
