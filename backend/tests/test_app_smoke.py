"""
End-to-end smoke test: import the FastAPI app and hit the /health route.

Goal: catch import-time regressions and config errors before they ship.
"""
from __future__ import annotations

import pytest


def test_app_imports_cleanly():
    """If any module has a circular import or bad type, this catches it."""
    from app.main import app  # noqa: F401
    assert app is not None


@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/health")
    assert r.status_code in (200, 204)


@pytest.mark.asyncio
async def test_recommendation_path_protected():
    """The recommendation endpoint must require auth, not 500."""
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/v1/coach/recommendation")
    assert r.status_code in (401, 403)
