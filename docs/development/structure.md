# Project Structure

The AI Imaging Agent is organized into modular components with clear separation of concerns.

## Directory Layout

```
ai-agent/
├── .github/
│   └── workflows/          # CI/CD workflows
│       └── deploy_docs.yml # Documentation deployment
├── artifacts/
│   └── rag_index/          # FAISS index and embeddings
├── dataset/
│   └── catalog.jsonl       # Software catalog
├── docs/                   # MkDocs documentation
├── logs/                   # Application logs
├── src/
│   └── ai_agent/          # Main package
│       ├── agent/         # PydanticAI agent
│       ├── api/           # Pipeline orchestration
│       ├── catalog/       # Catalog management
│       ├── generator/     # VLM selection (schemas)
│       ├── retriever/     # Text retrieval
│       ├── ui/            # Gradio interface
│       └── utils/         # Shared utilities
├── tests/                 # Test suite
├── config.yaml            # Model configuration
├── mkdocs.yml            # Documentation config
├── pyproject.toml        # Package metadata
└── README.md             # Project readme
```

## Core Modules

### src/ai_agent/

Main package containing all application code.

#### agent/

PydanticAI conversational agent implementation.

```
agent/
├── __init__.py
├── agent.py               # Agent definition, tools
├── state.py               # ChatState dataclass
└── tools.py               # Agent tools (search, repo_info)
```

**Key components**:
- `agent.py`: Agent instance, system prompt, tool definitions
- `state.py`: Conversation state management
- `tools.py`: Tool implementations (search_alternative, repo_info, etc.)

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

Software catalog synchronization.

```
catalog/
├── __init__.py
└── sync.py                # Catalog sync logic
```

**Functions**:
- Load catalog from JSONL
- Check for changes (SHA1)
- Trigger index rebuild

**Dependencies**: `retriever/`

#### generator/

VLM selection schemas and types.

```
generator/
├── __init__.py
└── schema.py              # Pydantic models for responses
```

**Models**:
- `ToolRecommendation`: Individual tool recommendation
- `AgentResponse`: Complete response with status
- `ConversationStatus`: Enum for conversation states
- `ToolReason`: Enum for recommendation reasons

**Dependencies**: None (pure schemas)

#### retriever/

Text-based retrieval pipeline.

```
retriever/
├── __init__.py
├── text_embedder.py       # BGE-M3 embedding model
├── vector_index.py        # FAISS index management
├── reranker.py            # CrossEncoder reranking
└── software_doc.py        # Catalog schema and loading
```

**Pipeline flow**:
1. `text_embedder.py`: Embed query
2. `vector_index.py`: FAISS search
3. `reranker.py`: CrossEncoder reranking
4. Output: Top-K candidates

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
- `tags.py`: Parse `[EXCLUDE:...]`, `[NO_RERANK]`, etc.

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
version = "0.1.1"
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
from ai_agent.agent.state import ChatState
```

**Never use** relative imports like `from ..utils import ...`

## Extension Points

### Adding New Tools

Add to `agent/tools.py`:

```python
@agent.tool
async def new_tool(ctx: RunContext[ChatState], param: str) -> str:
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
