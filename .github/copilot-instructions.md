# AI Agent — Copilot Instructions

This is a **RAG + VLM imaging tool recommender** that helps users find the right imaging software for their images and tasks. Users drop an image, describe their task, and get ranked software recommendations with demo links.

## Architecture Overview

The system follows a two-stage pipeline:

1. **Retrieval Stage** (`retriever/`, `api/pipeline.py`): Fast text search using BGE-M3 embeddings + CrossEncoder reranker. No LLM calls. Returns top-K candidates.

2. **Selection Stage** (`generator/`): Single VLM call (OpenAI GPT-4o/mini) that sees the image + candidates + metadata and returns ranked recommendations with accuracy scores.

### Key Components

- **`api/pipeline.RAGImagingPipeline`**: Main orchestrator. Handles file validation, metadata extraction, retrieval, and VLM selection.
- **`retriever/text_embedder.py`**, **`retriever/vector_index.py`**, **`retriever/reranker.py`**, **`retriever/software_doc.py`**: Embedding, FAISS indexing, reranking, and catalog schema for retrieval.
- **`agent/agent.py`**: PydanticAI agent that orchestrates tool search, alternatives, and recommendation assembly.
- **`utils/image_meta.py`**: Robust metadata extraction for DICOM, NIfTI, TIFF stacks with medical imaging focus.
- **`utils/tags.py`**: Control tag parsing/stripping utilities (notably `[EXCLUDE:tool1|tool2]`).

## Data Flow Patterns

### Input Processing
- Files validated via `utils/file_validator.py` (size limits, format checks)
- Images converted to PNG previews for VLM via `utils/previews.py`
- Metadata extracted preserving original format info (critical for format compatibility matching)
- Format tokens added to retrieval query (e.g. `format:DICOM format:NIfTI`)

### Retrieval Query Construction
```python
# Clean task text + format tokens from uploaded files
query = f"{clean_task} format:{ext_tokens}"  # e.g. "segment lungs format:DICOM"
```

### VLM Selection Input
The VLM receives:
- **Text**: User task + candidate table + original file metadata
- **Image**: PNG preview (converted from any format) 
- **Metadata**: Original extension, dimensions, file info (crucial for IO compatibility)

## Critical Patterns

### Error Handling
- **Graceful degradation**: If image conversion fails, continue text-only
- **Robust metadata**: All metadata extraction wrapped in try/catch with sensible defaults
- **File validation**: Early validation prevents downstream errors

### Control Tags System
Users can control behavior via tags in their queries:
- `[EXCLUDE:toolname1|toolname2]` - Exclude specific tools from results

### Conversation Flow
- **Complete**: Normal success with tool recommendations
- **Needs Clarification**: VLM asks followup questions when task is ambiguous
- **Terminal No-Tool**: No suitable tools found with explanation

## Development Workflows

### Running the App
```bash
# Install with pip using pyproject.toml
pip install -e ".[dev]"

# Configure .env with OPENAI_API_KEY and SOFTWARE_CATALOG path
ai_agent chat  # Launches Gradio chat UI
```

### Testing
- Run targeted tests in `tests/` (e.g., retrieval, agent tools, repo info)
- Run with: `pytest tests/`

### Change Documentation
- **`CHANGELOG.md`**: Follow [Keep a Changelog](https://keepachangelog.com/) format
- Use semantic versioning with sections: Added, Changed, Deprecated, Removed, Fixed, Security
- Update CHANGELOG.md for ALL user-facing changes before merging PRs
- Format: `### Added\n- New feature description` under version heading
- Version entries: `## [x.y.z] - YYYY-MM-DD`

### Environment Management
- **uv**: Fast Python package manager used in `tools/image/Dockerfile`
- Creates isolated `.venv` environments for reproducible builds
- Dockerfile uses `uv venv && uv pip install -e .` pattern for container builds

### Logging & Debugging
- Set `LOG_PROMPTS=1` to save VLM prompts + images to `logs/`
- File logs in `logs/app_YYYYMMDD.log` with structured JSON events
- Console/file log levels configurable via `.env`

## Project Conventions

### Schema Patterns
- **Pydantic models** in `generator/schema.py` with robust field validation and aliasing for catalog compatibility
- **Enum-based** conversation states and tool reasons for type safety
- **Field normalization**: Dimensions (2D/3D/4D), modalities (CT/MRI/XR), file formats via validators

### Catalog Integration
- Software catalog in JSONL format following schema.org SoftwareSourceCode structure
- **Runnable examples**: Links to HuggingFace Spaces, notebooks, web demos
- **Supporting data**: Format compatibility info used for matching

### Module Boundaries
- `api/`: Pipeline orchestration, no UI dependencies
- `generator/`: Pure VLM logic, no retrieval dependencies  
- `retriever/`: Pure vector search, no generation dependencies
- `utils/`: Shared utilities, no business logic
- `ui/`: Gradio interface only

### Configuration
- Environment-based config via `.env` (API keys, model names, catalog paths)
- Sensible defaults for all settings
- No hardcoded paths or credentials

## Medical Imaging Context

This tool specializes in medical/scientific imaging:
- **Modalities**: CT, MRI, X-ray, Ultrasound, PET, SPECT, Microscopy
- **Formats**: DICOM, NIfTI, TIFF stacks, standard images
- **Dimensions**: 2D images, 3D volumes, 4D timeseries
- **Tasks**: Segmentation, registration, analysis, visualization

The VLM selection considers format compatibility as a primary factor - tools supporting the user's input format are strongly preferred.

## Security Notes
- Only makes external calls to OpenAI VLM API (with user image preview)
- Never uploads user data to third-party tool demos
- Returns links only; user chooses whether to visit demos
- Prompt logging is optional and local-only