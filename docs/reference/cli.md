# CLI Commands

The AI Imaging Agent exposes three command modes through both `ai_agent` and `ai-agent`.

## `ai_agent serve`

Start the FastAPI backend used by the React frontend.

```bash
ai_agent serve
```

What it does:

1. Runs a startup catalog sync attempt.
2. Starts background catalog refresh when `SYNC_EVERY_HOURS` is set.
3. Initializes the shared retrieval pipeline on FastAPI startup.
4. Serves API routes under `/api/*`.
5. Serves the built React SPA when `FRONTEND_DIST_DIR` contains `index.html`.

Default local URL:

```text
http://localhost:8000
```

Configurable environment variables:

```dotenv
HOST=0.0.0.0
PORT=8000
FRONTEND_DIST_DIR=src/frontend/dist
DEV_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Development frontend:

```bash
cd src/frontend
npm run dev
```

Production frontend:

```bash
cd src/frontend
npm run build
cd ../..
ai_agent serve
```

## `ai_agent chat`

Launch the legacy Gradio UI.

```bash
ai_agent chat
```

This mode still performs startup catalog sync, initializes the retrieval pipeline, and starts background refresh. Use it for legacy UI testing or fallback workflows. New frontend work should target `ai_agent serve` plus `src/frontend`.

## `ai_agent sync`

Run one catalog refresh without starting a UI or API server.

```bash
ai_agent sync
```

What it does:

1. Queries the configured GraphDB SPARQL endpoint.
2. Saves the raw JSON-LD snapshot.
3. Converts catalog data to JSONL.
4. Detects changes via SHA-1.
5. Rebuilds FAISS index artifacts when the catalog changes.

Common sync variables:

```dotenv
GRAPHDB_URL=https://graphdb.example.com/repositories/imaging
GRAPHDB_GRAPH=https://example.org/graph/imaging-tools
GRAPHDB_QUERY_FILE=src/ai_agent/queries/get_relevant_software.rq
OUTPUT_JSONLD=dataset/catalog.jsonld
OUTPUT_JSONL=dataset/catalog.jsonl
SYNC_FORCE=0
```

Force sync:

```bash
SYNC_FORCE=1 ai_agent sync
```

## Command Aliases

```bash
ai_agent serve
ai-agent serve

ai_agent chat
ai-agent chat

ai_agent sync
ai-agent sync
```

## Common Workflows

### React Development

Terminal 1:

```bash
ai_agent serve
```

Terminal 2:

```bash
cd src/frontend
npm run dev
```

Open `http://localhost:5173`.

### Docker

```bash
docker build -t ai-agent .
docker run -p 7860:7860 --env-file .env ai-agent
```

Docker runs `ai_agent serve` and exposes the bundled frontend on `http://localhost:7860`.

### Debug Logging

```bash
LOGLEVEL_CONSOLE=DEBUG LOG_PROMPTS=1 ai_agent serve
```

## Troubleshooting

### Command Not Found

```bash
pip install -e .
python -m ai_agent.cli serve
```

### Backend Port Already In Use

Set a different port:

```bash
PORT=8010 ai_agent serve
```

If using Vite, update the proxy target:

```bash
cd src/frontend
VITE_API_TARGET=http://localhost:8010 npm run dev
```

### Frontend Shows API Errors

- Check that `ai_agent serve` is running.
- Check `DEV_CORS_ORIGINS` if the frontend origin changed.
- Check `APP_PASSWORD` and sign in again if auth is enabled.

### Catalog Load Error

```bash
ls -lh dataset/catalog.jsonl
python -c "import json; [json.loads(l) for l in open('dataset/catalog.jsonl')]"
```

## Next Steps

- Configure [Environment Variables](environment.md)
- Return to [Quick Start](../getting-started/quickstart.md)
