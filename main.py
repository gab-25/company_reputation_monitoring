from fastapi import FastAPI
from pydantic import BaseModel

from sentiment_analysis import analyze

app = FastAPI()


class SentimentBody(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/sentiment")
async def sentiment(body: SentimentBody):
    return analyze(body.text)
