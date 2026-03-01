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

## Required Configuration

### OpenAI API Key

The AI Imaging Agent requires an OpenAI API key for the vision-language model:

1. Sign up for an account at [OpenAI](https://platform.openai.com/)
2. Navigate to [API Keys](https://platform.openai.com/api-keys)
3. Create a new API key
4. Add it to your `.env` file:

```dotenv
OPENAI_API_KEY=sk-your-actual-key-here
```

## Model Configuration

The agent model can be configured via `config.yaml`:

```yaml
# AI Agent Model Configuration

# Default/fallback model (used for CLI and initial startup)
agent_model:
  name: "gpt-4o-mini"
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
  
  - display_name: "gpt-5.1"
    name: "gpt-5.1"
    base_url: null
    provider: "OpenAI"
    api_key_env: "OPENAI_API_KEY"
```

### Using Alternative Model Providers

You can configure custom OpenAI-compatible endpoints:

```yaml
available_models:
  - display_name: "EPFL Inference"
    name: "gpt-4o-mini"
    base_url: "https://inference.epfl.ch/v1"
    provider: "EPFL"
    api_key_env: "EPFL_API_KEY"
```

Then add the corresponding API key to your `.env`:

```dotenv
EPFL_API_KEY=your-epfl-key
```

## Optional Configuration

### GitHub Token

For the repository info tool (optional):

```dotenv
GITHUB_TOKEN=ghp_your_github_personal_access_token
```

This enables the agent to fetch detailed information about GitHub repositories.

### Pipeline Parameters

Adjust retrieval and recommendation settings:

```dotenv
# Number of candidate tools to retrieve
TOP_K=8

# Number of final recommendations to show
NUM_CHOICES=3
```

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
