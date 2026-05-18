from __future__ import annotations

from backend.app.core.exceptions import AuthenticationError
from backend.app.core.security import create_token, hash_password, verify_password
from backend.app.core.settings import Settings
from backend.app.repositories.user_repository import UserRepository
from backend.app.schemas.auth import Role, TokenPair, UserInDB, UserPublic

ACCESS_TOKEN_TYPE = "access"  # nosec B105
REFRESH_TOKEN_TYPE = "refresh"  # nosec B105


class AuthService:
    def __init__(self, settings: Settings, user_repository: UserRepository):
        self.settings = settings
        self.user_repository = user_repository

    async def bootstrap_default_user(self) -> None:
        role = Role.ADMIN if self.settings.default_admin_role == "admin" else Role.USER
        admin = UserInDB(
            username=self.settings.default_admin_username,
            role=role,
            disabled=False,
            hashed_password=hash_password(self.settings.default_admin_password),
        )
        await self.user_repository.upsert(admin)

    async def get_user(self, username: str) -> UserPublic | None:
        user = await self.user_repository.get(username)
        if user is None:
            return None
        return UserPublic(username=user.username, role=user.role, disabled=user.disabled)

    async def authenticate(self, username: str, password: str) -> UserPublic:
        user = await self.user_repository.get(username)
        if user is None or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid username or password")
        if user.disabled:
            raise AuthenticationError("User is disabled")
        return UserPublic(username=user.username, role=user.role, disabled=user.disabled)

    def issue_tokens(self, user: UserPublic) -> TokenPair:
        access_token = create_token(
            username=user.username,
            role=user.role,
            secret_key=self.settings.secret_key,
            algorithm=self.settings.jwt_algorithm,
            ttl_minutes=self.settings.access_token_ttl_minutes,
            token_type=ACCESS_TOKEN_TYPE,
        )
        refresh_token = create_token(
            username=user.username,
            role=user.role,
            secret_key=self.settings.secret_key,
            algorithm=self.settings.jwt_algorithm,
            ttl_minutes=self.settings.refresh_token_ttl_minutes,
            token_type=REFRESH_TOKEN_TYPE,
        )
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.settings.access_token_ttl_minutes * 60,
        )
