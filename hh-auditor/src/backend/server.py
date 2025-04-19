from fastapi import FastAPI
from src.backend.agent import stream

app = FastAPI()


@app.get("/")
async def read_root():
    return stream()
