from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    USER = "user"
    ADMIN = "admin"


class UserPublic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    role: Role
    disabled: bool = False


class UserInDB(UserPublic):
    hashed_password: str


class TokenPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(ge=1)


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str = Field(min_length=1)
