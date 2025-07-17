import uvicorn

from fastapi import FastAPI
from contextlib import asynccontextmanager

from database.connection import database
from settings import settings

async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}