# Configuration

Before running the AI Imaging Agent, you need to configure it with your API keys and preferences.

## Environment Variables

Create a `.env` file in the repository root with the following configuration:

```dotenv
# Required: OpenAI API key
OPENAI_API_KEY=sk-xxxx

# Optional: GitHub token for repository info tool
GITHUB_TOKEN=ghp_xxxx

# Optional: Alternative model providers
EPFL_API_KEY=sk-xxxx

# Software catalog path
SOFTWARE_CATALOG=dataset/catalog.jsonl

# Logging configuration
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0         # Set to 1 to save prompt snapshots for debugging

# Custom config path
CONFIG_PATH=config.yaml
```

## Required Configuration

### API Key

The AI Imaging Agent requires an API key for the vision-language model. Which key you need depends on the `agent_model` in `config.yaml`:

**For the default EPFL endpoint** (`config.yaml` default):

```dotenv
EPFL_API_KEY=your-epfl-key
EPFL_API_KEY_EMBEDDER=your-epfl-embedder-key
```

**For a standard OpenAI endpoint**:

1. Sign up for an account at [OpenAI](https://platform.openai.com/)
2. Navigate to [API Keys](https://platform.openai.com/api-keys)
3. Create a new API key
4. Add it to your `.env` file:

```dotenv
OPENAI_API_KEY=sk-your-actual-key-here
```

## Model Configuration

The agent model and retrieval stack are configured via `config.yaml`. The defaults use the EPFL OpenAI-compatible inference endpoint:

```yaml
# AI Agent Model Configuration

# Default model (used for CLI and initial startup)
agent_model:
  name: "openai/gpt-oss-120b"
  base_url: "https://inference-rcp.epfl.ch/v1"
  api_key_env: "EPFL_API_KEY"

# Available models for UI dropdown
available_models:
  - display_name: "gpt-4o-mini"
    name: "gpt-4o-mini"
    base_url: null          # null = standard OpenAI endpoint
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"
  
  - display_name: "gpt-4o"
    name: "gpt-4o"
    base_url: null
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"

  - display_name: "openai/gpt-oss-120b [EPFL]"
    name: "openai/gpt-oss-120b"
    base_url: "https://inference-rcp.epfl.ch/v1"
    provider: "EPFL"
    api_key_env: "EPFL_API_KEY"

# Retrieval stack (embedder + reranker)
retrieval:
  embedder:
    backend: "remote"
    model_name: "Qwen/Qwen3-Embedding-8B"
    base_url: "https://inference-rcp.epfl.ch/v1"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
    timeout_s: 20

  reranker:
    backend: "remote"
    model_name: "BAAI/bge-reranker-v2-m3"
    base_url: "https://inference-rcp.epfl.ch/v1"
    api_key_env: "EPFL_API_KEY_EMBEDDER"
    timeout_s: 20
```

### Using Standard OpenAI Models

To use standard OpenAI models instead of the EPFL endpoint, update `agent_model` in `config.yaml`:

```yaml
agent_model:
  name: "gpt-4o-mini"
  base_url: null                        # null = default OpenAI endpoint
  api_key_env: "OPENAI_API_KEY"
```

Then add `OPENAI_API_KEY` to your `.env`.

### Using Alternative Model Providers

Any OpenAI-compatible endpoint can be configured:

```yaml
agent_model:
  name: "your-model-name"
  base_url: "https://your-endpoint.example.com/v1"
  api_key_env: "YOUR_CUSTOM_API_KEY"
```

Then add the corresponding API key to your `.env`:

```dotenv
YOUR_CUSTOM_API_KEY=your-key
```

### Local Retrieval (No Remote Embedder)

To run the embedder and reranker locally (no remote endpoint needed):

```yaml
retrieval:
  embedder:
    backend: "local"
    model_name: "BAAI/bge-m3"
  reranker:
    backend: "local"
    model_name: "BAAI/bge-reranker-v2-m3"
```

Install the required extras:

```bash
pip install sentence-transformers
```

## Optional Configuration

### GitHub Token

For the repository info tool (optional):

```dotenv
GITHUB_TOKEN=ghp_your_github_personal_access_token
```

This enables the agent to fetch detailed information about GitHub repositories via direct API calls. Without it, the tool falls back to DeepWiki MCP or the repocards library.

### Pipeline Parameters

Adjust recommendations via environment variables:

```dotenv
NUM_CHOICES=3    # Number of tool recommendations (default: 3)
```

You can also override these values per-session from the UI settings panel.

### Catalog Sync

The `ai_agent sync` command (and background auto-refresh) requires a GraphDB SPARQL endpoint:

```dotenv
GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
GRAPHDB_QUERY_FILE=get_relevant_software.rq
SYNC_EVERY_HOURS=24     # 0 to disable background refresh
```

See [Environment Variables](../reference/environment.md) for the full list.

### Logging

Configure logging behavior:

```dotenv
# Console log level (DEBUG, INFO, WARNING, ERROR)
LOGLEVEL_CONSOLE=WARNING

# File log level
LOGLEVEL_FILE=INFO

# Enable file logging (0 or 1)
FILE_LOG=1

# Log directory
LOG_DIR=logs

# Save VLM prompts and images for debugging (0 or 1)
LOG_PROMPTS=0
```

!!! tip "Debug Mode"
    Set `LOG_PROMPTS=1` to save VLM prompts and images to the `logs/` directory. This is useful for debugging but will increase disk usage.

### Software Catalog

Specify the path to your software catalog:

```dotenv
SOFTWARE_CATALOG=dataset/catalog.jsonl
```

The catalog should be in JSONL format following the schema.org SoftwareSourceCode structure.

## Verification

After configuring, verify your setup:

```bash
# Check that environment variables are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

## Next Steps

With configuration complete, you're ready to:

- [Run the Quick Start](quickstart.md)
- Learn about [Using the Chat Interface](../user-guide/chat-interface.md)
