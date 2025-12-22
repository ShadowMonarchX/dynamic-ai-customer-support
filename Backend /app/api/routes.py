from fastapi import APIRouter, HTTPException

router = APIRouter()

fake_users_db = [{"username": "john_doe"}, {"username": "jane_smith"}]

@router.get("/")
def read_users():
    return fake_users_db

@router.get("/{username}")
def read_user(username: str):
    for user in fake_users_db:
        if user["username"] == username:
            return user
    raise HTTPException(status_code=404, detail="User not found")

@router.post("/")
def create_user(user: dict):
    fake_users_db.append(user)
    return user
