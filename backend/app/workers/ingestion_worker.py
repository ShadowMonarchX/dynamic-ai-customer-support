from __future__ import annotations

import asyncio

from backend.app.core.settings import get_settings
from backend.app.services.ingestion_service import IngestionService
from backend.app.workers.celery_app import celery_app


@celery_app.task(name="workers.rebuild_vector_artifacts")
def rebuild_vector_artifacts() -> int:
    settings = get_settings()
    ingestion = IngestionService(settings)
    return asyncio.run(ingestion.rebuild_artifacts())
