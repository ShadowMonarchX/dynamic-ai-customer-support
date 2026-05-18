from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from backend.app.core.security import require_roles
from backend.app.schemas.auth import Role, UserPublic
from backend.app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    user: Annotated[UserPublic, Depends(require_roles({Role.USER, Role.ADMIN}))],
) -> QueryResponse:
    _ = user
    session_id = payload.session_id or request.state.session_id
    trace_id = request.state.trace_id
    return await request.app.state.container.orchestration_service.process_query(
        query=payload.user_query,
        session_id=session_id,
        trace_id=trace_id,
    )


@router.post("/stream")
async def query_stream(
    payload: QueryRequest,
    request: Request,
    user: Annotated[UserPublic, Depends(require_roles({Role.USER, Role.ADMIN}))],
) -> StreamingResponse:
    _ = user
    session_id = payload.session_id or request.state.session_id

    async def event_stream():
        async for token in request.app.state.container.orchestration_service.stream_query(
            query=payload.user_query,
            session_id=session_id,
        ):
            yield f"data: {token}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
