# Project Structure

The AI Imaging Agent is organized into modular components with clear separation of concerns.

## Directory Layout

```
ai-agent/
├── .github/
│   └── copilot-instructions.md  # Architecture + agent instructions
├── artifacts/
│   └── rag_index/               # FAISS index and metadata
│       ├── index.faiss
│       └── meta.json
├── dataset/
│   ├── catalog.jsonl            # Software catalog (JSONL)
│   ├── catalog.jsonld           # Raw JSON-LD from SPARQL fetch
│   └── catalog.jsonl.sha1       # SHA-1 for change detection
├── docs/                        # MkDocs documentation source
├── logs/                        # Application logs
├── src/
│   └── ai_agent/               # Main package
│       ├── agent/              # PydanticAI agent + tools
│       ├── api/                # Pipeline orchestration
│       ├── catalog/            # Catalog sync (SPARQL → JSONL)
│       ├── core/               # Shared pipeline singleton
│       ├── generator/          # VLM schemas and prompts
│       ├── queries/            # SPARQL query files
│       ├── retriever/          # Embedding, FAISS, reranking
│       ├── ui/                 # Gradio interface
│       └── utils/              # Shared utilities
├── tests/                      # Test suite
├── config.yaml                 # Model and retrieval configuration
├── mkdocs.yml                  # Documentation config
├── pyproject.toml              # Package metadata, dependencies
└── README.md                   # Project readme
```

## Core Modules

### src/ai_agent/

Main package containing all application code.

#### agent/

PydanticAI conversational agent implementation.

```
agent/
├── __init__.py
├── agent.py               # Agent definition, tool registration, run_agent()
├── models.py              # Agent output/log models
├── utils.py               # AgentState, limit_tool_calls() decorator
└── tools/
    ├── __init__.py
    ├── deepwiki_tool.py       # Repository info via DeepWiki MCP
    ├── gradio_space_tool.py   # Gradio Space demo discovery
    ├── query_utils.py         # Query preprocessing helpers
    ├── repo_info_tool.py      # repo_info_batch (GitHub + DeepWiki)
    ├── search_alternative_tool.py  # search_alternative tool
    ├── search_tool.py         # search_tools primary tool
    ├── utils.py               # Shared tool helpers
    └── mcp/                   # MCP server integrations
        ├── __init__.py
        ├── base.py
        ├── lungs_segmentation_tool.py
        └── registry.py
```

**Key components**:

- `agent.py`: Agent instance, system prompt, `run_agent()` entry point
- `models.py`: Agent output and tool usage schemas
- `utils.py`: `AgentState` model, `limit_tool_calls()` prepare hook
- `tools/`: Modular tool implementations (search, alternatives, repo info, MCP)

**Dependencies**: `api/`, `utils/`

#### api/

Pipeline orchestration and core logic.

```
api/
├── __init__.py
└── pipeline.py            # RAGImagingPipeline main class
```

**Responsibilities**:

- File validation and metadata extraction
- Retrieval + VLM selection orchestration
- Error handling and logging
- Index management

**Dependencies**: `retriever/`, `generator/`, `utils/`

#### catalog/

Software catalog synchronization via SPARQL.

```
catalog/
├── __init__.py
└── sync.py                # sync_once() — SPARQL fetch → JSON-LD → JSONL → FAISS
```

**Functions**:

- Fetch catalog from GraphDB SPARQL endpoint
- Convert JSON-LD to JSONL with SHA-1 change detection
- Trigger FAISS index rebuild when catalog changes
- `sync_once()`: one-shot sync; runs at startup and on schedule

**Dependencies**: `retriever/`

#### core/

Shared pipeline singleton used across CLI, UI, and tools.

```
core/
├── __init__.py
└── pipeline_registry.py   # Singleton get_pipeline() / reset_pipeline()
```

**Key function**: `get_pipeline()` returns the shared `RAGImagingPipeline` instance, initializing it on first call.

**Dependencies**: `api/`

#### queries/

SPARQL query files used by the catalog sync.

```
queries/
└── get_relevant_software.rq   # SPARQL query with {graph} placeholder
```

The query file path can be overridden via the `GRAPHDB_QUERY_FILE` environment variable.

#### generator/

VLM selection schemas and types.

```
generator/
├── __init__.py
└── schema.py              # Pydantic models for responses
```

**Models**:

- `ToolSelection`: Selected tool with accuracy score
- `ToolChoice`: Individual recommendation
- `Conversation`: Full conversation output with status
- `ConversationStatus`: Enum (complete / needs_clarification / no_tool)
- `ToolReason`: Enum for recommendation reasons

**Dependencies**: None (pure schemas)

#### retriever/

Text-based retrieval pipeline.

```
retriever/
├── __init__.py
├── text_embedder.py       # LocalBGEEmbedder — remote (Qwen3-Embedding-8B) or local
├── vector_index.py        # FAISS IndexFlatIP management
├── reranker.py            # CrossEncoderReranker — remote (BGE-M3) or local
└── software_doc.py        # SoftwareDoc schema and catalog loading
```

**Pipeline flow**:

1. `text_embedder.py`: Embed query with Qwen3-Embedding-8B (remote by default)
2. `vector_index.py`: FAISS exact inner-product search
3. `reranker.py`: BGE-M3 CrossEncoder reranking (disabled if API key missing)
4. Output: Top-K candidates with relevance scores

**Dependencies**: None (pure retrieval)

#### ui/

Gradio web interface.

```
ui/
├── __init__.py
├── app.py                 # Gradio app definition
├── components.py          # Reusable UI components
├── formatters.py          # Response formatting
├── handlers.py            # Message handlers
├── state.py               # UI state management
└── visualizations.py      # Preview and trace rendering
```

**Key files**:

- `app.py`: Main Gradio interface
- `handlers.py`: `respond()` function - core interaction logic
- `formatters.py`: Format recommendations as markdown/cards
- `components.py`: Reusable Gradio components

**Dependencies**: `agent/`, `api/`

#### utils/

Shared utilities.

```
utils/
├── __init__.py
├── config.py              # Configuration loading
├── file_validator.py      # File validation
├── image_meta.py          # Metadata extraction (DICOM, NIfTI, TIFF)
├── previews.py            # Image preview generation
└── tags.py                # Control tag parsing
```

**Common utilities**:

- `config.py`: Load `config.yaml` with Pydantic validation
- `file_validator.py`: Size limits, format checks
- `image_meta.py`: Extract DICOM/NIfTI/TIFF metadata
- `previews.py`: Convert medical images to PNG
- `tags.py`: Parse exclusion tags and strip control tags from queries

**Dependencies**: None (pure utilities)

#### cli.py

Command-line interface entry point.

```python
def main():
    # Parse arguments
    # Route to chat or sync
```

**Commands**:

- `ai_agent chat`: Launch UI
- `ai_agent sync`: Sync catalog

### tests/

Test suite.

```
tests/
├── data/
│   └── test_data.json     # Test cases
├── test_retrieval_pipeline.py
├── test_deepwiki_repo_info.py
└── ...
```

**Test categories**:

- Unit tests: Individual components
- Integration tests: Full pipeline
- End-to-end tests: Real API calls (optional)

## Configuration Files

### pyproject.toml

Python package metadata and dependencies.

```toml
[project]
name = "ai_agent"
version = "1.0.0"
dependencies = [...]

[project.scripts]
ai_agent = "ai_agent.cli:main"
```

### config.yaml

Model configuration.

```yaml
agent_model:
  name: "gpt-4o-mini"
  base_url: null
  api_key_env: "OPENAI_API_KEY"

available_models:
  - display_name: "gpt-4o-mini"
    name: "gpt-4o-mini"
    ...
```

### mkdocs.yml

Documentation configuration.

```yaml
site_name: AI Imaging Agent
theme:
  name: material
nav: [...]
```

### .env

Environment variables (not committed).

```dotenv
OPENAI_API_KEY=sk-xxxx
SOFTWARE_CATALOG=dataset/catalog.jsonl
```

## Data Files

### dataset/catalog.jsonl

Software catalog in JSON Lines format.

Each line is a complete JSON object following schema.org SoftwareSourceCode.

### artifacts/rag_index/

Pre-built FAISS index and metadata.

```
artifacts/rag_index/
├── index.faiss            # FAISS binary index
└── meta.json              # Tool IDs, config, timestamps
```

## Module Boundaries

Clear separation prevents circular dependencies:

```
ui/ → agent/ → api/ → retriever/
                  → generator/
                  → utils/
```

**Rules**:

- `utils/`: No dependencies on other modules
- `retriever/`: Pure retrieval, no generation
- `generator/`: Pure schemas, no retrieval
- `api/`: Orchestrates retriever + generator
- `agent/`: Uses api for tool calls
- `ui/`: Top-level, depends on agent + api

## Import Patterns

All imports use absolute paths from `ai_agent`:

```python
from ai_agent.retriever.vector_index import VectorIndex
from ai_agent.utils.config import load_config
from ai_agent.agent.utils import AgentState
```

**Never use** relative imports like `from ..utils import ...`

## Extension Points

### Adding New Tools

Add tool adapters to `agent/agent.py` and implement logic in `agent/tools/`:

```python
@agent.tool
async def new_tool(ctx: RunContext[AgentState], param: str) -> str:
    """Tool description."""
    # Implementation
    return result
```

### Adding New Metadata Extractors

Add to `utils/image_meta.py`:

```python
def extract_custom_format(file_path: str) -> dict:
    """Extract metadata from custom format."""
    # Implementation
    return metadata
```

### Adding New Retrieval Models

Replace in `retriever/text_embedder.py`:

```python
class TextEmbedder:
    def __init__(self, model_name="new-embedding-model"):
        self.model = SentenceTransformer(model_name)
```

## Next Steps

- Learn about [Contributing](contributing.md)
- Explore [Testing](testing.md)
- Return to [Architecture Overview](../architecture/overview.md)
