from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import uvicorn
import asyncio


class Item(BaseModel):
    text: str


app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(item: Item):
    return classifier(item.text)[0]


# Запуск сервера Uvicorn
async def main():
    config = uvicorn.Config("main:app", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
