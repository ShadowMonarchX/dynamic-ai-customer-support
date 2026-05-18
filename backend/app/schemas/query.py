from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.contracts import ComplexityLevel, IntentLabel, UrgencyLevel


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    score: float = Field(ge=0.0, le=1.0)
    excerpt: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    intent: IntentLabel
    urgency: UrgencyLevel
    complexity: ComplexityLevel
    citations: list[Citation]
    trace_id: str
    session_id: str
