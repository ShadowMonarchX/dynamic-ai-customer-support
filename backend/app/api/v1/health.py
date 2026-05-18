from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok",
        "service": request.app.state.container.settings.app_name,
        "environment": request.app.state.container.settings.environment,
    }
