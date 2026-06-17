"""FastAPI app factory.

Mounts all routers under ``/api/*`` plus (in production) serves the built
frontend bundle from ``FRONTEND_DIST_DIR`` (default ``src/frontend/dist``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ai_agent.agent.tools import ensure_tools_registered
from ai_agent.api.deps import get_pipeline
from ai_agent.api.routers import auth, catalog, chat, files, health, models

log = logging.getLogger("api.server")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Imaging Agent",
        version="2.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # CORS — permissive in dev so Vite (5173) can call :8000. In prod the
    # frontend is served from the same origin, so this is a no-op.
    dev_origins = os.getenv(
        "DEV_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in dev_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(models.router)
    app.include_router(catalog.router)
    app.include_router(files.router)
    app.include_router(chat.router)

    @app.on_event("startup")
    async def _startup() -> None:
        ensure_tools_registered()
        # Touch the pipeline once so first request doesn't pay the boot cost.
        try:
            pipe = get_pipeline()
            log.info("Pipeline ready with %d docs", len(pipe.index.docs))
        except Exception:
            log.exception("Pipeline initialization failed")

    # Production: serve the React bundle. In dev, Vite serves it.
    dist_dir = Path(os.getenv("FRONTEND_DIST_DIR") or "src/frontend/dist")
    if dist_dir.exists() and (dist_dir / "index.html").exists():
        # Static assets first (so JS/CSS hits return immediately)
        app.mount("/assets", StaticFiles(directory=dist_dir / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            # Never shadow /api/* — FastAPI matches routers first, but be explicit.
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "not_found"}, status_code=404)
            candidate = dist_dir / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist_dir / "index.html")

        log.info("Serving frontend from %s", dist_dir)
    else:
        log.info(
            "No frontend bundle at %s — assuming dev mode (Vite on :5173)", dist_dir
        )

    return app


app = create_app()


__all__ = ["app", "create_app"]
