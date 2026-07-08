# Multi-stage build:
#   1. Build the Vite/React frontend
#   2. Install and run the FastAPI backend

# ---- Stage 1: frontend build ----
FROM node:20-alpine AS frontend-build

WORKDIR /app

COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY src/frontend ./
RUN npm run build


# ---- Stage 2: Python runtime ----
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 user

ENV HOME=/home/user \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/home/user/.local/bin:$PATH

# Create a dedicated Python environment.
RUN python -m venv "$VIRTUAL_ENV" && \
    python -m pip install --upgrade pip

WORKDIR /home/user/app

# Copy application source as root because installation also runs as root.
COPY . .

# Copy the compiled frontend.
COPY --from=frontend-build /app/dist ./src/frontend/dist

# Remove generated host metadata, then install into /opt/venv.
RUN rm -rf src/*.egg-info && \
    python -m pip install --no-cache-dir . && \
    chown -R user:user /home/user/app

# Drop privileges only after installation.
USER user

EXPOSE 7860

ENV PORT=7860 \
    HOST=0.0.0.0 \
    FRONTEND_DIST_DIR=src/frontend/dist

CMD ["ai_agent", "serve"]