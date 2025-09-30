FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Crear usuario no-root con UID/GID que suele usar VS Code (1000:1000)
RUN useradd -ms /bin/bash -u 1000 vscode \
    && apt-get update && apt-get install -y sudo \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER non-root-user
WORKDIR /workspaces

# Copy only dependency files first (for better caching)
COPY --chown=non-root-user:non-root-user pyproject.toml ./

# Create venv and install dependencies
RUN uv venv && \
    echo '. $PWD/.venv/bin/activate' >> /home/non-root-user/.bashrc

# Copy the rest of the project
COPY --chown=non-root-user:non-root-user . ./

# Install the package in editable mode
RUN source .venv/bin/activate && \
    uv pip install -e .

# Set default shell to use the virtual environment
SHELL ["/bin/bash", "-c"]
EXPOSE 7860
ENTRYPOINT ["/bin/bash", "-c", "source .venv/bin/activate && python -m ai_agent.app"]