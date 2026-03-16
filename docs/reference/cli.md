# CLI Commands

The AI Imaging Agent provides a command-line interface for starting the application and managing the software catalog.

## Available Commands

### ai_agent chat

Launch the chat-based user interface.

```bash
ai_agent chat
```

**What it does**:

1. Performs startup catalog synchronization
2. Loads the FAISS index
3. Initializes the retrieval and agent pipelines
4. Launches the Gradio web interface on `http://127.0.0.1:7860`
5. Starts background catalog refresh (if configured)

**Options**: None (all configuration via `.env` and `config.yaml`)

**Example**:

```bash
$ ai_agent chat
[startup-sync] 150 → dataset/catalog.jsonl
[startup-refresh] catalog unchanged; keeping existing FAISS index
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

**Background Refresh**:

If `SYNC_EVERY_HOURS` is set in `.env`, the catalog will auto-refresh in the background:

```dotenv
SYNC_EVERY_HOURS=24  # Check every 24 hours
```

### ai_agent sync

Manually synchronize the software catalog and rebuild the index.

```bash
ai_agent sync
```

**What it does**:

1. Loads the software catalog from `SOFTWARE_CATALOG` path
2. Embeds all tool descriptions using BGE-M3
3. Builds FAISS vector index
4. Saves artifacts to `artifacts/rag_index/`

**When to use**:

- After editing `catalog.jsonl`
- After adding new tools
- To force index rebuild
- For testing catalog changes

**Example**:

```bash
$ ai_agent sync
[sync] 150 → dataset/catalog.jsonl
[sync] Embedding 150 tools... (5.2s)
[sync] Building FAISS index...
[sync] Saved to artifacts/rag_index/
[sync] Sync complete.
```

## Command Aliases

Both commands are available with either `ai_agent` or `ai-agent`:

```bash
ai_agent chat   # Works
ai-agent chat   # Also works

ai_agent sync   # Works
ai-agent sync   # Also works
```

## Common Usage Patterns

### Development Workflow

```bash
# Edit catalog
vim dataset/catalog.jsonl

# Sync catalog
ai_agent sync

# Test changes
ai_agent chat
```

<!-- ### Production Deployment

```bash
# In your deployment script:
ai_agent sync                    # Ensure index is built
nohup ai_agent chat &           # Run in background
```

Or use environment variable control:

```bash
export SYNC_EVERY_HOURS=0       # Disable auto-refresh in production
ai_agent chat
``` -->

### Testing & Development

```bash
# Enable debug logging
export LOGLEVEL_CONSOLE=DEBUG
export LOG_PROMPTS=1
ai_agent chat
```

## Environment Variables

All configuration is via environment variables (see [Environment Variables Reference](environment.md)).

## Exit Codes

- **0**: Success
- **1**: General error (see logs)

## Troubleshooting

### Command Not Found

If you see `command not found: ai_agent`:

```bash
# Ensure package is installed
pip install -e .

# Check installation
pip list | grep ai-agent

# Try with python -m
python -m ai_agent.cli chat
```

### Port Already in Use

If port 7860 is occupied:

```bash
# Find and kill process
lsof -ti:7860 | xargs kill -9

# Or change port in code (ui/app.py)
```

### Catalog Load Error

If catalog fails to load:

```bash
# Verify catalog exists
ls -lh dataset/catalog.jsonl

# Verify JSONL syntax
python -c "import json; [json.loads(l) for l in open('dataset/catalog.jsonl')]"

# Check environment variable
echo $SOFTWARE_CATALOG
```

### Index Build Error

If FAISS index building fails:

```bash
# Check artifacts directory
ls -lh artifacts/rag_index/

# Rebuild manually
rm -rf artifacts/rag_index/
ai_agent sync
```

## Next Steps

- Configure [Environment Variables](environment.md)
- Review the [Changelog](changelog.md)
- Return to [Getting Started](../getting-started/quickstart.md)
