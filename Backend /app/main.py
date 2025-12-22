from fastapi import FastAPI
from routers import items, users  # Import the routers

app = FastAPI(
    title="My Awesome API",
    description="This is a simple API built with FastAPI",
    version="1.0.0",
)


app.include_router(items.router, prefix="/items", tags=["items"])
app.include_router(users.router, prefix="/users", tags=["users"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}
