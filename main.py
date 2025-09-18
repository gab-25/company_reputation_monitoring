from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()


class AnalizeBody(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analize")
async def sentiment(body: AnalizeBody):
    """
    Analizza il testo e restituisce un dizionario con le classifiche.
    """
    # Inizializza la pipeline con il tuo modello addestrato
    # sentiment_analyzer = pipeline("sentiment-analysis", model="Gab-25/company_reputation")
    sentiment_analyzer = pipeline("sentiment-analysis", model="./results/checkpoint-465")

    # Esegui una previsione
    result = sentiment_analyzer(body.text)

    # Il risultato sarà una lista di dizionari con label e score
    return result
