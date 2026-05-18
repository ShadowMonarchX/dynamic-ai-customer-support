from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntentLabel(str, Enum):
    GREETING = "greeting"
    IDENTITY = "identity"
    SERVICE_QUERY = "service_query"
    CONTACT_REQUEST = "contact_request"
    TRANSACTIONAL = "transactional"
    FAQ = "faq"
    GENERAL = "general"
    UNKNOWN = "unknown"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ComplexityLevel(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    BIG = "big"


class RetrievalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    text: str
    source: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docs: list[RetrievalDocument] = Field(default_factory=list)
    count: int = 0
    status: str = "empty"
    top_similarity: float = 0.0


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    trust_message: str
    relevance: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)


class IntentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentLabel
    urgency: UrgencyLevel
    complexity: ComplexityLevel
    emotion: str = "neutral"
    language: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    topic: str = "general"
