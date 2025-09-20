import os
import time

import torch
from fastapi import FastAPI, Response
from prometheus_client import generate_latest, Counter, Histogram, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from torch import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEV_MODE = os.getenv("DEV_MODE", "false") == "true"
print(f"DEV_MODE: {DEV_MODE}")

app = FastAPI()

# Definisci le metriche
REQUEST_COUNT = Counter('model_requests_total', 'Total requests', ['status'])
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration')

# Carica il modello e il tokenizer
model_name = "Gab-25/company_reputation" if DEV_MODE == False else "./results/checkpoint-465"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"load model {model} on {device}")


class AnalizeBody(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analize")
async def analize(body: AnalizeBody):
    """
    Analizza il testo e restituisce un dizionario con le classifiche.
    """
    start_time = time.time()
    try:
        # Tokenizza il testo
        inputs = tokenizer(body.text, return_tensors="pt", truncation=True, padding=True).to(device)

        # Esegue la predizione senza calcolo del gradiente
        with torch.no_grad():
            outputs = model(**inputs)

        # Ottiene i logits (output grezzo prima dell'attivazione)
        logits = outputs.logits

        # Applica softmax per convertire i logits in probabilità
        probabilities = softmax(logits, dim=1)[0]  # Ottiene le probabilità per il primo (e unico) elemento nel batch

        # Prepara il risultato, includendo i punteggi per tutte le etichette per completezza
        all_scores = []
        for i, score in enumerate(probabilities):
            all_scores.append({"label": model.config.id2label[i], "score": score.item()})

        REQUEST_COUNT.labels(status='success').inc()

        return all_scores
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        raise
    finally:
        REQUEST_DURATION.observe(time.time() - start_time)


@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
