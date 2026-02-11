# AI Imaging Agent (Imaging Plaza)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

An intelligent RAG + AI agent system that helps users discover the right imaging software for their images and tasks. Upload an image, describe what you want to do, and get ranked tool recommendations with links to runnable demos.

## ✨ Key Features

- **🤖 Conversational AI Agent**: Natural language interaction with multi-turn context
- **🔍 Smart Retrieval**: BGE-M3 embeddings + FAISS + CrossEncoder reranking
- **👁️ Vision-Aware Selection**: VLM-based tool selection considering both image content and metadata
- **🏥 Medical Imaging Focus**: Specialized support for CT, MRI, DICOM, NIfTI, and other medical formats
- **🎯 Format-Aware Matching**: IO compatibility scoring based on file formats and dimensions
- **🚀 Demo Integration**: Direct execution of Gradio Space demos on your images
- **📊 Rich UI**: Chat interface with image previews, file management, and execution traces

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10–3.12
- OpenAI API key (or compatible API endpoint)
- Internet connection for model calls

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install the package
pip install --upgrade pip
pip install -e .

# For development (includes test dependencies)
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file at the repository root:

```dotenv
# Required: OpenAI API key
OPENAI_API_KEY=sk-xxxx

# Optional: GitHub token for repo info tool
GITHUB_TOKEN=ghp_xxxx

# Optional: Alternative model providers (EPFL, etc.)
EPFL_API_KEY=sk-xxxx

# Software catalog path
SOFTWARE_CATALOG=dataset/catalog.jsonl

# Pipeline configuration
TOP_K=8                # Number of candidates to retrieve
NUM_CHOICES=3          # Number of tools to recommend

# Logging configuration
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0         # Set to 1 to save prompt snapshots for debugging

# Custom config path
CONFIG_PATH=config.yaml
```

### Model Configuration

The agent model can be configured via `config.yaml`:

```yaml
# AI Agent Model Configuration

# Default/fallback model (used for CLI and initial startup)
agent_model:
  name: "gpt-5.1"
  base_url: null                        # null for default OpenAI endpoint
  api_key_env: "OPENAI_API_KEY"

# Available models for UI dropdown
available_models:
  - display_name: "gpt-4o-mini"
    name: "gpt-4o-mini"
    base_url: null
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"
  
  - display_name: "gpt-4o"
    name: "gpt-4o"
    base_url: null
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"
  
  - display_name: "gpt-5-mini"
    name: "gpt-5-mini"
    base_url: null
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"

  - display_name: "gpt-5.1"
    name: "gpt-5.1"
    base_url: null
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"
```

### Running the App

```bash
# Start the chat interface
ai_agent chat

# Open your browser to:
# http://127.0.0.1:7860
```

Try uploading a cat image and asking:
> "I want to segment the cat from this image"

---

## 💬 Usage

### Chat Interface

The chat interface provides a natural conversation flow:

1. **Upload Files**: Drop images (PNG, JPG, TIFF, DICOM, NIfTI, etc.) or other supported files
2. **Describe Your Task**: Use natural language like "segment the lungs" or "register brain MRI"
3. **Review Recommendations**: Get ranked tool suggestions with accuracy scores and explanations
4. **Run Demos**: Click "Run demo" to execute tools directly on your uploaded images
5. **Iterate**: Ask for alternatives, refine your query, or upload different files

### Supported File Formats

**Images:**
- Standard: PNG, JPG, JPEG, WEBP, BMP, GIF
- Medical: DICOM (.dcm), NIfTI (.nii, .nii.gz), TIFF stacks
- Scientific: Multi-page TIFF, TIFF with metadata

**Other Files:**
- Data: CSV, JSON, XML
- Media: MP3, MP4

### Example Queries

- "Segment the lungs from this CT scan"
- "Register these two brain MRI images"
- "Extract text from this medical report image"
- "Classify what organ is shown in this ultrasound"
- "Detect tumors in this MRI scan"
- "I need to analyze DICOM files, what tools are available?"

### Understanding Results

Each recommendation includes:
- **Rank**: Priority order (1 = best match)
- **Accuracy Score**: Confidence level (0-100%)
- **Explanation**: Why this tool matches your request
- **Metadata**: Supported modalities, dimensions, formats, license
- **Demo Link**: Direct link to runnable example

---

## 🏗️ Architecture

### Pipeline Overview

The system follows a two-stage architecture:

```
User Input (Image + Text Query)
        ↓
┌───────────────────────────────┐
│   RETRIEVAL STAGE             │
│  - BGE-M3 Embeddings          │
│  - FAISS Vector Search        │
│  - CrossEncoder Reranking     │
│  - Format Token Matching      │
└───────────────────────────────┘
        ↓ Top-K Candidates
┌───────────────────────────────┐
│   AGENT SELECTION             │
│  - Pydantic AI Agent          │
│  - OpenAI VLM                 │
│  - Image + Metadata Analysis  │
│  - Multi-Tool Reasoning       │
└───────────────────────────────┘
        ↓
   Ranked Recommendations
```

### Retrieval Stage

**No LLM calls** - purely text-based search:

1. **Query Construction**: User task + format tokens from uploaded files
2. **Embedding**: BGE-M3 model generates query embedding
3. **Vector Search**: FAISS retrieves top candidates
4. **Reranking**: CrossEncoder refines results for precision
5. **Similarity Expansion**: Automatic query enrichment with semantic neighbors

### Agent Selection Stage

**Single VLM call** - multimodal reasoning:

1. **Input Preparation**:
   - Text: User query + candidate table + file metadata
   - Image: PNG preview (converted from any format)
   - Context: Original file format, dimensions, modality

2. **Agent Tools**:
   - `search_tools`: Search catalog with query
   - `search_alternative`: Find alternatives (iterative)
   - `run_example`: Execute Gradio Space demos
   - `repo_summary`: Fetch GitHub documentation via DeepWiki MCP

3. **Output**: Ranked tool selections with accuracy scores and explanations

### Key Components

- **`api/pipeline.py`**: RAG retrieval orchestrator
- **`agent/agent.py`**: Pydantic AI agent with tool definitions
- **`retriever/`**: Embedding, FAISS indexing, reranking
- **`generator/`**: Prompts and schema for tool selection
- **`ui/`**: Gradio chat interface components
- **`utils/`**: Image processing, metadata extraction, file validation
- **`catalog/`**: Catalog syncing from GraphDB (optional)

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `GITHUB_TOKEN` | GitHub token for repo info | - | ❌ |
| `SOFTWARE_CATALOG` | Path to catalog JSONL | `dataset/catalog.jsonl` | ✅ |
| `TOP_K` | Retrieval candidates count | `8` | ❌ |
| `NUM_CHOICES` | Tools to recommend | `3` | ❌ |
| `LOGLEVEL_CONSOLE` | Console log level | `WARNING` | ❌ |
| `LOGLEVEL_FILE` | File log level | `INFO` | ❌ |
| `FILE_LOG` | Enable file logging | `1` | ❌ |
| `LOG_DIR` | Log directory | `logs` | ❌ |
| `LOG_PROMPTS` | Save prompt snapshots | `0` | ❌ |
| `CONFIG_PATH` | Model config file | `config.yaml` | ✅ |

### GraphDB Catalog Sync (Optional)

For automatic catalog syncing from a GraphDB instance:

```dotenv
GRAPHDB_URL=https://your-graphdb.example.com
GRAPHDB_GRAPH=your-graph-name
GRAPHDB_USER=username
GRAPHDB_PASSWORD=password
GRAPHDB_QUERY_FILE=/path/to/query.rq
SYNC_EVERY_HOURS=24  # Auto-refresh interval (0 to disable)
OUTPUT_JSONLD=dataset/catalog.jsonld
OUTPUT_JSONL=dataset/catalog.jsonl
```

Run manual sync:
```bash
ai_agent sync
```

---

## 📋 Catalog Format

The catalog is a JSONL file where each line is a `SoftwareDoc` following schema.org SoftwareSourceCode structure.

### Minimal Example

```json
{
  "name": "3d-lungs-segmentation",
  "description": "3D lung segmentation from CT; returns a mask/overlay.",
  
  "applicationCategory": ["Medical Imaging"],
  "featureList": ["segmentation"],
  "imagingModality": ["CT"],
  "dims": [3],
  "anatomy": ["lung"],
  "keywords": ["mask", "overlay", "lung segmentation", "CT"],
  
  "programmingLanguage": "Python",
  "requiresGPU": false,
  "isAccessibleForFree": true,
  "license": "Apache-2.0",
  
  "supportingData": [
    {
      "datasetFormat": "TIFF",
      "bodySite": "lung",
      "imagingModality": "CT",
      "hasDimensionality": 3
    },
    {
      "datasetFormat": "DICOM",
      "bodySite": "lung",
      "imagingModality": "CT",
      "hasDimensionality": 3
    }
  ],
  
  "runnableExample": [
    {
      "hostType": "gradio",
      "url": "https://huggingface.co/spaces/qchapp/3d-lungs-segmentation",
      "name": "HF Space"
    }
  ]
}
```

### Key Fields

- **name**: Unique identifier for the tool
- **description**: Clear explanation of what the tool does
- **featureList**: Operations (e.g., segmentation, registration, classification)
- **imagingModality**: Medical imaging types (CT, MRI, XR, US, PET)
- **dims**: Supported dimensions (2D, 3D, 4D)
- **anatomy**: Body parts/organs
- **supportingData**: Format compatibility information (critical for matching)
- **runnableExample**: Links to live demos (HuggingFace Spaces, notebooks, web apps)

---

## 🔧 Development

### Project Structure

```
ai-agent/
├── src/ai_agent/
│   ├── agent/              # Pydantic AI agent and tools
│   │   ├── agent.py        # Agent definition
│   │   ├── models.py       # Agent state models
│   │   ├── tools/          # Agent tool implementations
│   │   │   ├── search_tool.py
│   │   │   ├── search_alternative_tool.py
│   │   │   ├── gradio_space_tool.py
│   │   │   ├── repo_info_tool.py
│   │   │   └── deepwiki_tool.py
│   │   └── utils.py
│   ├── api/                # Pipeline orchestration
│   │   └── pipeline.py     # RAGImagingPipeline
│   ├── retriever/          # Retrieval components
│   │   ├── text_embedder.py
│   │   ├── vector_index.py
│   │   ├── reranker.py
│   │   └── software_doc.py
│   ├── generator/          # Agent prompts and schemas
│   │   ├── prompts.py
│   │   └── schema.py
│   ├── ui/                 # Gradio interface
│   │   ├── app.py
│   │   ├── handlers.py
│   │   ├── components.py
│   │   ├── formatters.py
│   │   ├── state.py
│   │   └── visualizations.py
│   ├── utils/              # Shared utilities
│   │   ├── config.py       # Configuration management
│   │   ├── file_validator.py
│   │   ├── image_meta.py   # Metadata extraction
│   │   ├── image_io.py
│   │   ├── previews.py
│   │   └── tags.py
│   ├── catalog/            # Catalog syncing
│   │   └── sync.py
│   └── cli.py              # CLI entry point
├── tests/                  # Test suite
│   ├── test_retrieval_pipeline.py
│   ├── test_repo_summary.py
│   └── data/
├── artifacts/              # Generated artifacts
│   └── rag_index/          # FAISS index
├── dataset/                # Catalog data
│   └── catalog.jsonl
├── logs/                   # Application logs
├── config.yaml             # Model configuration
├── pyproject.toml          # Project metadata & dependencies
├── Dockerfile              # Production Docker image
├── tools/image/Dockerfile  # Development Docker image
└── justfile                # Task runner commands
```

### Local Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_retrieval_pipeline.py

# With verbose output
pytest -v tests/

# With coverage
pytest --cov=ai_agent tests/
```

### Logging & Debugging

**Console Logs**: Set `LOGLEVEL_CONSOLE=DEBUG` for verbose output

**File Logs**: Automatically saved to `logs/app_YYYYMMDD.log` (rotates daily)

**Prompt Snapshots**: Enable `LOG_PROMPTS=1` to save:
- `logs/vlm_selector_YYYYMMDD_HHMMSS.txt` - System/user prompts

---

## 📚 API & CLI Reference

### CLI Commands

```bash
# Launch chat interface
ai_agent chat

# Sync catalog from GraphDB
ai_agent sync
```

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Recent Highlights

**[Unreleased]**
- ✨ New chat-based interface with conversational AI
- 🔧 YAML model configuration for flexible deployment
- 🔍 Similarity-based query expansion (no hardcoded synonyms)
- 🔗 DeepWiki MCP integration for GitHub repo docs
- 🎨 Imaging Plaza branding and custom theme
- 🛠️ Agent-based architecture replacing legacy VLM selector
- ⚡ Iterative retrieval with automatic retry logic
- 🗑️ Removed deprecated code paths and outdated tests

**[0.1.3] - 2025-10-22**
- Gradio space runner tool
- Repository info tool
- UI fixes and polish

---

## 📄 License

No license specified for now.

---

## 🙏 Credits & Acknowledgments

**Developed by**: Imaging Plaza Team

**Technologies:**
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) - AI agent framework
- [OpenAI](https://openai.com) - GPT vision model
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multilingual embeddings
- [Gradio](https://gradio.app) - Interactive web UI
- [DeepWiki](https://deepwiki.com) - GitHub repository documentation

**Medical Imaging Formats:**
- [pydicom](https://github.com/pydicom/pydicom) - DICOM support
- [nibabel](https://nipy.org/nibabel/) - NIfTI support

---

## 📮 Support

For issues, questions, or contributions, please contact the Imaging Plaza team.

---

**🏥 Medical Disclaimer**: This software is a tool recommendation system, not a diagnostic tool. Always consult qualified medical professionals for clinical decisions.