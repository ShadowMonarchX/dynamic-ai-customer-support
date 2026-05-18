from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from passlib.context import CryptContext

from backend.app.core.exceptions import AuthenticationError, AuthorizationError
from backend.app.schemas.auth import Role, UserPublic

# pbkdf2_sha256 avoids environment-specific bcrypt backend issues while
# remaining a strong password hashing scheme for service authentication.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_token(
    *,
    username: str,
    role: Role,
    secret_key: str,
    algorithm: str,
    ttl_minutes: int,
    token_type: str,
) -> str:
    now = datetime.now(tz=UTC)
    payload = {
        "sub": username,
        "role": role.value,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_token(token: str, *, secret_key: str, algorithm: str) -> dict:
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
    except InvalidTokenError as exc:
        raise AuthenticationError("Invalid or expired token") from exc

    username = payload.get("sub")
    token_type = payload.get("type")
    role = payload.get("role")
    if not username or not token_type or not role:
        raise AuthenticationError("Malformed token payload")
    return payload


async def get_current_user(
    request: Request,
    token: Annotated[str, Depends(oauth2_scheme)],
) -> UserPublic:
    settings = request.app.state.container.settings
    claims = decode_token(token, secret_key=settings.secret_key, algorithm=settings.jwt_algorithm)
    if claims.get("type") != "access":
        raise AuthenticationError("Invalid token type")
    user = await request.app.state.container.auth_service.get_user(claims["sub"])
    if not user:
        raise AuthenticationError("User not found")
    if user.disabled:
        raise AuthenticationError("User is disabled")
    return user


def require_roles(allowed_roles: set[Role]) -> Callable:
    async def _role_guard(user: Annotated[UserPublic, Depends(get_current_user)]) -> UserPublic:
        if user.role not in allowed_roles:
            raise AuthorizationError()
        return user

    return _role_guard
