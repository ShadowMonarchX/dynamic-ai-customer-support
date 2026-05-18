from __future__ import annotations

import asyncio

from backend.app.core.settings import get_settings
from backend.app.services.ingestion_service import IngestionService


async def _run() -> None:
    settings = get_settings()
    service = IngestionService(settings)
    count = await service.rebuild_artifacts()
    print(f"Rebuilt {count} vector artifacts at {settings.vector_artifact_path}")


if __name__ == "__main__":
    asyncio.run(_run())
