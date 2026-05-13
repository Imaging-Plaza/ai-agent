# Multi-stage build:
#   1. node:20-alpine compiles the Vite/React frontend at src/frontend
#   2. python:3.11-slim installs the package and runs the FastAPI backend,
#      which also serves the built bundle from /home/user/app/src/frontend/dist.

# ---- Stage 1: frontend build ----
FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY src/frontend ./
RUN npm run build

# ---- Stage 2: python runtime ----
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user . .
COPY --from=frontend-build --chown=user /app/dist ./src/frontend/dist

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 7860
ENV PORT=7860 \
    HOST=0.0.0.0 \
    FRONTEND_DIST_DIR=src/frontend/dist

# Run the FastAPI backend (which also serves the SPA). To fall back to the
# legacy Gradio UI, override CMD: `docker run ... ai_agent chat`.
CMD ["ai_agent", "serve"]
