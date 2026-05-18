from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm

from backend.app.core.exceptions import AuthenticationError
from backend.app.core.security import decode_token
from backend.app.schemas.auth import RefreshRequest, TokenPair

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenPair)
async def issue_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
) -> TokenPair:
    user = await request.app.state.container.auth_service.authenticate(
        username=form_data.username,
        password=form_data.password,
    )
    return request.app.state.container.auth_service.issue_tokens(user)


@router.post("/refresh", response_model=TokenPair)
async def refresh_token(payload: RefreshRequest, request: Request) -> TokenPair:
    settings = request.app.state.container.settings
    claims = decode_token(
        payload.refresh_token,
        secret_key=settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )
    if claims.get("type") != "refresh":
        raise AuthenticationError("Invalid token type")

    user = await request.app.state.container.auth_service.get_user(claims["sub"])
    if user is None:
        raise AuthenticationError("User not found")

    return request.app.state.container.auth_service.issue_tokens(user)
