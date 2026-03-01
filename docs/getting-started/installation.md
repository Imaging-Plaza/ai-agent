# Installation

This guide will help you install and set up the AI Imaging Agent on your system.

## Prerequisites

Before installing, ensure you have:

- **Python 3.10–3.12** installed
- **pip** (Python package manager)
- **OpenAI API key** (or compatible API endpoint)
- Internet connection for model calls

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/imaging-plaza/ai-agent.git
cd ai-agent
```

### 2. Create Virtual Environment

It's recommended to use a virtual environment to isolate dependencies:

=== "Linux/macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows"

    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

### 3. Install the Package

For regular use:

```bash
pip install --upgrade pip
pip install -e .
```

For development (includes test dependencies):

```bash
pip install -e ".[dev]"
```

## Verify Installation

Verify that the installation was successful:

```bash
ai_agent --help
```

You should see the available commands:

```
usage: ai_agent [-h] {chat,sync}

AI Agent CLI

positional arguments:
  {chat,sync}  'chat' launches the chat UI; 'sync' runs one catalog refresh.
```

## Next Steps

Now that you have installed the AI Imaging Agent, proceed to:

- [Configuration](configuration.md) - Set up your environment and API keys
- [Quick Start](quickstart.md) - Run your first query

## Troubleshooting

### Python Version Issues

If you encounter issues with Python version compatibility:

```bash
# Check your Python version
python --version

# Use a specific Python version
python3.10 -m venv .venv
```

### Installation Errors

If you encounter dependency conflicts:

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Try installing again
pip install -e .
```

### Missing System Dependencies

Some packages may require system libraries:

=== "Ubuntu/Debian"

    ```bash
    sudo apt-get update
    sudo apt-get install python3-dev build-essential
    ```

=== "macOS"

    ```bash
    # Using Homebrew
    brew install python@3.10
    ```

=== "Windows"

    Ensure you have [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed.

## Docker Installation (Alternative)

A Dockerfile is available for containerized deployment:

```bash
# Build the Docker image
docker build -t ai-agent -f tools/image/Dockerfile .

# Run the container
docker run -p 7860:7860 --env-file .env ai-agent
```

!!! note
    Make sure to create a `.env` file with your configuration before running the Docker container.
