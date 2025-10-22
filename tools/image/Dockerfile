FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Set default shell to bash for all subsequent RUN commands
SHELL ["/bin/bash", "-c"]

# Create non-root user with UID/GID typically used by VS Code (1000:1000)
RUN useradd -ms /bin/bash -u 1000 non-root-user \
    && apt-get update && apt-get install -y sudo \
    && echo "non-root-user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER non-root-user
WORKDIR /workspaces

# Copy only dependency files first (for better caching)
COPY --chown=non-root-user:non-root-user pyproject.toml ./

# Create venv and install dependencies
RUN uv venv && \
    echo '. $PWD/.venv/bin/activate' >> /home/non-root-user/.bashrc

# Copy the rest of the project
COPY --chown=non-root-user:non-root-user . ./

# Install the package in editable mode (now using bash by default)
RUN source .venv/bin/activate && uv pip install -e .

EXPOSE 7860
ENTRYPOINT ["source .venv/bin/activate && python -m ai_agent.app"]