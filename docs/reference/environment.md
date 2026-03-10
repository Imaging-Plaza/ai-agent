# Environment Variables

Configuration for the AI Imaging Agent is managed via environment variables, typically defined in a `.env` file.

## Required Variables

### OPENAI_API_KEY

OpenAI API key for vision-language model calls.

```dotenv
OPENAI_API_KEY=sk-xxxx
```

**Where to get it**: [OpenAI API Keys](https://platform.openai.com/api-keys)

**Required**: Yes (unless using alternative model provider)

**Used by**: Agent VLM calls, tool selection

## Optional Variables

### SOFTWARE_CATALOG

Path to the software catalog JSONL file.

```dotenv
SOFTWARE_CATALOG=dataset/catalog.jsonl
```

**Default**: `dataset/catalog.jsonl`

**Required**: No (uses default)

### TOP_K

Number of candidate tools to retrieve from FAISS search.

```dotenv
TOP_K=8
```

**Default**: `8`

**Range**: 1-50 (recommended: 5-10)

**Impact**: More candidates = better recall but slower VLM calls

### NUM_CHOICES

Number of final tool recommendations to return to user.

```dotenv
NUM_CHOICES=3
```

**Default**: `3`

**Range**: 1-10 (recommended: 3-5)

**Impact**: Too many recommendations can overwhelm users

### GITHUB_TOKEN

GitHub personal access token for repository info tool.

```dotenv
GITHUB_TOKEN=ghp_xxxx
```

**Where to get it**: [GitHub Tokens](https://github.com/settings/tokens)

**Permissions needed**: `public_repo` (read access)

**Required**: No (tool gracefully degrades without it)

**Benefits**: Higher API rate limits, access to private repos

### SYNC_EVERY_HOURS

Auto-refresh catalog interval in hours.

```dotenv
SYNC_EVERY_HOURS=24
```

**Default**: `0` (disabled)

**Range**: 0 (disabled) or ≥1

**Behavior**: Background thread checks catalog every N hours and rebuilds index if changed

## Logging Configuration

### LOGLEVEL_CONSOLE

Console logging level.

```dotenv
LOGLEVEL_CONSOLE=WARNING
```

**Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Default**: `WARNING`

**Recommendation**: 
- Development: `DEBUG` or `INFO`
- Production: `WARNING`

### LOGLEVEL_FILE

File logging level.

```dotenv
LOGLEVEL_FILE=INFO
```

**Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Default**: `INFO`

**Files written to**: `logs/app_YYYYMMDD.log`

### FILE_LOG

Enable file logging.

```dotenv
FILE_LOG=1
```

**Options**: `0` (disabled), `1` (enabled)

**Default**: `1`

### LOG_DIR

Directory for log files.

```dotenv
LOG_DIR=logs
```

**Default**: `logs`

**Created automatically** if it doesn't exist

### LOG_PROMPTS

Save VLM prompts and images for debugging.

```dotenv
LOG_PROMPTS=1
```

**Options**: `0` (disabled), `1` (enabled)

**Default**: `0`

**Writes to**: `logs/prompts/YYYYMMDD_HHMMSS/`

**Contents**:
- `prompt.txt`: Text prompt sent to VLM
- `image_*.png`: Images included in prompt
- `response.json`: VLM response
- `metadata.json`: Request metadata

**Warning**: Can consume significant disk space over time

## Model Configuration

### CONFIG_PATH

Path to YAML model configuration file.

```dotenv
CONFIG_PATH=config.yaml
```

**Default**: `config.yaml`

**See**: `config.yaml` for model configuration details

### Alternative Model Providers

For custom OpenAI-compatible endpoints, configure in `config.yaml`:

```yaml
agent_model:
  name: "model-name"
  base_url: "https://api.example.com/v1"
  api_key_env: "CUSTOM_API_KEY"
```

Then in `.env`:

```dotenv
CUSTOM_API_KEY=your-key-here
```

<!-- ## Advanced Configuration

### RERANK_TOP_N

Number of candidates to retrieve before reranking.

```dotenv
RERANK_TOP_N=20
```

**Default**: `20`

**Interaction with TOP_K**:
- FAISS retrieves `RERANK_TOP_N` candidates (e.g., 20)
- CrossEncoder reranks them
- Top `TOP_K` (e.g., 8) passed to VLM

**Recommendation**: 2-3x `TOP_K` value

### SIMILARITY_THRESHOLD

Minimum cosine similarity for query expansion.

```dotenv
SIMILARITY_THRESHOLD=0.75
```

**Default**: `0.75`

**Range**: 0.0-1.0

**Impact**: Lower = more expansion terms (broader search)

### MAX_EXPANSION_TERMS

Maximum number of terms to add during query expansion.

```dotenv
MAX_EXPANSION_TERMS=10
```

**Default**: `10`

**Range**: 0-50

**Impact**: More terms = broader search but potential noise -->

## .env File Example

Complete example `.env` file:

```dotenv
# Required
OPENAI_API_KEY=sk-xxxx

# Optional: Alternative providers
EPFL_API_KEY=sk-xxxx
GITHUB_TOKEN=ghp_xxxx

# Catalog
SOFTWARE_CATALOG=dataset/catalog.jsonl
SYNC_EVERY_HOURS=24

# Pipeline
TOP_K=8
NUM_CHOICES=3

# Logging
LOGLEVEL_CONSOLE=WARNING
LOGLEVEL_FILE=INFO
FILE_LOG=1
LOG_DIR=logs
LOG_PROMPTS=0  # Set to 1 for debugging

# Model configuration
CONFIG_PATH=config.yaml
```

## Loading Environment Variables

### Automatic Loading

The application automatically loads `.env` from the repository root:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env automatically
```

### Manual Loading

```bash
# Export manually
export OPENAI_API_KEY=sk-xxxx
export TOP_K=8

# Or source .env
set -a
source .env
set +a
```

### Docker

Pass environment variables to Docker:

```bash
docker run --env-file .env ai-agent
```

## Security Best Practices

!!! warning "Never commit .env files"
    Add `.env` to `.gitignore` to prevent accidental commits

!!! warning "Protect API keys"
    Treat API keys as sensitive credentials:
    - Never share in public repositories
    - Rotate keys if exposed
    - Use environment-specific keys (dev/prod)

!!! tip "Use .env.example"
    Create `.env.example` with dummy values for documentation:
    
    ```dotenv
    OPENAI_API_KEY=sk-your-key-here
    GITHUB_TOKEN=ghp-your-token-here
    ```

## Next Steps

- Review [CLI Commands](cli.md)
- Check [Configuration Guide](../getting-started/configuration.md)
- See [Changelog](changelog.md)
