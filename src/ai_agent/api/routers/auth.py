"""Password login routes.

The frontend POSTs ``{password}`` to ``/api/auth/login``; if the value
matches ``APP_PASSWORD`` (env), an HMAC token is set as an httpOnly cookie
and all subsequent ``/api/*`` requests authenticate via that cookie.

When ``APP_PASSWORD`` is unset, login still works (any password accepted)
and the cookie is not strictly required — useful for local dev or when the
deployment sits behind a Cloudflare tunnel without per-user auth.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Response, status

from ai_agent.api.deps import (
    AUTH_COOKIE_NAME,
    auth_disabled,
    make_cookie_value,
    verify_password,
)
from ai_agent.api.schemas import LoginRequest, LoginResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/status")
def status_route() -> dict:
    """Tells the frontend whether auth is enforced at all.

    Used by the login page to skip itself when ``APP_PASSWORD`` is empty.
    """
    return {"required": not auth_disabled()}


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, response: Response) -> LoginResponse:
    if not verify_password(body.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_password"
        )
    expected_pw = os.getenv("APP_PASSWORD") or ""
    if expected_pw:
        response.set_cookie(
            key=AUTH_COOKIE_NAME,
            value=make_cookie_value(expected_pw),
            httponly=True,
            samesite="lax",
            secure=False,  # tunnel terminates TLS; cookie travels over plain HTTP internally
            max_age=60 * 60 * 24 * 7,
            path="/",
        )
    return LoginResponse(ok=True)


@router.post("/logout", response_model=LoginResponse)
def logout(response: Response) -> LoginResponse:
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return LoginResponse(ok=True)
