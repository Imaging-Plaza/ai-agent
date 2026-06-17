# Installation

This guide installs the Python backend and the React frontend.

## Prerequisites

- Python 3.10-3.12
- Node.js 20+ and npm
- OpenAI API key or another OpenAI-compatible endpoint key
- Internet access for model calls and optional catalog sync

The dev container uses Python 3.12 and `uv`; local installs can use either `uv` or `pip`.

## Clone The Repository

```bash
git clone https://github.com/imaging-plaza/ai-agent.git
cd ai-agent
```

## Python Backend

### Dev Container / uv

```bash
uv venv
uv pip install -e .
uv pip install -e ".[dev]"
```

### Local pip

=== "Linux/macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[dev]"
    ```

=== "Windows"

    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    pip install --upgrade pip
    pip install -e ".[dev]"
    ```

## React Frontend

```bash
cd src/frontend
npm install
npm run build
cd ../..
```

For development, use `npm run dev` instead of `npm run build`.

## Verify Installation

```bash
ai_agent --help
```

Expected modes:

```text
usage: ai_agent [-h] {chat,sync,serve}

AI Agent CLI

positional arguments:
  {chat,sync,serve}
    chat   launches the legacy Gradio UI
    sync   runs one catalog refresh
    serve  starts the FastAPI backend used by the React frontend
```

Check the frontend:

```bash
cd src/frontend
npm run lint
```

## Docker Installation

The root `Dockerfile` builds the Vite frontend and runs the FastAPI backend. The backend serves the built SPA from the same origin.

```bash
docker build -t ai-agent .
docker run -p 7860:7860 --env-file .env ai-agent
```

Open `http://localhost:7860`.

The included `docker-compose.yml` also starts a Cloudflare tunnel sidecar:

```bash
docker compose up --build
```

## Troubleshooting

### Python Version Issues

```bash
python --version
python3.12 -m venv .venv
```

### Backend Dependency Errors

```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

### Frontend Dependency Errors

```bash
cd src/frontend
npm ci
```

Use `npm install` when changing dependencies and `npm ci` when reproducing `package-lock.json`.

### Missing System Dependencies

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get update
    sudo apt-get install python3-dev build-essential git
    ```

=== "macOS"

    ```bash
    brew install python@3.12 node@20
    ```

=== "Windows"

    Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if native Python packages fail to build.

## Next Steps

- [Configuration](configuration.md)
- [Quick Start](quickstart.md)
