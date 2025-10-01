# Alternative approach if ai_agent ui doesn't support those flags

# Default host (can be overridden with HOST=value just serve)
HOST := env_var_or_default("HOST", "0.0.0.0")
PORT := env_var_or_default("PORT", "7860")

# Install dependencies
install:
    uv pip install .

# Install in development mode
dev-install:
    uv pip install -e ".[dev]"

# Serve the app in production mode
serve: install
    HOST={{HOST}} PORT={{PORT}} ai_agent ui

# Serve the app in development mode with file watching
serve-dev: dev-install
    HOST={{HOST}} PORT={{PORT}} GRADIO_RELOAD=1 ai_agent ui

# Development setup (alias for dev-install)
dev: dev-install

# List all available commands
default:
    @just --list