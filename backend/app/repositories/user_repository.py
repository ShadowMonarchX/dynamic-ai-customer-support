from __future__ import annotations

from backend.app.schemas.auth import UserInDB


class UserRepository:
    def __init__(self):
        self._users: dict[str, UserInDB] = {}

    async def get(self, username: str) -> UserInDB | None:
        return self._users.get(username)

    async def upsert(self, user: UserInDB) -> None:
        self._users[user.username] = user
