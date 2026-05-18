from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SessionIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        session_id = request.headers.get("x-session-id") or str(uuid.uuid4())
        request.state.session_id = session_id
        response: Response = await call_next(request)
        response.headers["x-session-id"] = session_id
        return response
