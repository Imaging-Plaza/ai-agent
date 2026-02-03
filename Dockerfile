# Dockerfile at repo root

# 1. Base image
FROM python:3.11-slim

# 2. (Optional but useful) system deps for building wheels etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git \
    && rm -rf /var/lib/apt/lists/*

# 3. Create non-root user as HF recommends
RUN useradd -m -u 1000 user
USER user

# 4. Basic env + working dir
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 5. Copy project code into the image
COPY --chown=user . .

# 6. Install Python deps + your package so the `ai_agent` CLI exists
#    If you normally do `pip install -e .` locally, this is the Docker equivalent.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# 7. Expose the port the app will listen on
EXPOSE 7860
ENV PORT=7860

# 8. Start agent
CMD ["ai_agent", "chat"]
