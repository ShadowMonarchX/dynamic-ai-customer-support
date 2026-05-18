from __future__ import annotations

from celery import Celery

from backend.app.core.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "dynamic_ai_customer_support",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
