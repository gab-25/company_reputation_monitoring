import argparse

import pandas as pd
import torch
from datasets import Dataset
from datasets.table import Table
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorWithPadding, pipeline

# Carica il modello e il tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels, arrow_table: Table):
        super().__init__(arrow_table)
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train():
    """
    addestramento modello dal dataset di sentiment labelled sentences
    """
    # Carica il dataset dai file txt
    dataframe = pd.DataFrame()
    files_txt = [
        "sentiment_labelled_sentences/amazon_cells_labelled.txt",
        "sentiment_labelled_sentences/imdb_labelled.txt",
        "sentiment_labelled_sentences/yelp_labelled.txt"
    ]
    for file in files_txt:
        df = pd.read_csv(file, sep='\t', names=['sentence', 'label'])
        # Pre - elaborazione molto semplice: Conversione in minuscolo
        df['sentence'] = df['sentence'].str.lower()
        dataframe = pd.concat([dataframe, df])

    # Converti il DataFrame in un oggetto Dataset
    dataset = Dataset.from_pandas(dataframe)

    # Dividi il dataset in set di addestramento e validazione
    split_dataset = dataset.train_test_split(test_size=0.1)

    # Definisci una funzione per tokenizzare il testo
    # Tokenizzare significa convertire il testo in un formato numerico che il modello può capire
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], truncation=True, max_length=128)

    tokenized_datasets = split_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Imposta gli argomenti per l'addestramento
    training_args = TrainingArguments(
        output_dir="./results",  # Directory dove salvare i checkpoint del modello
        num_train_epochs=3,  # Numero di epoche di addestramento
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,  # Numero di step di "riscaldamento" per il learning rate
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch"  # Valuta le performance alla fine di ogni epoca
    )

    # Inizializza l'oggetto Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator
    )

    # Avvia l'addestramento
    trainer.train()

    # Fai delle previsioni sul set di validazione
    predictions = trainer.predict(tokenized_datasets['test'])
    # Prendi la classe con la probabilità più alta (la previsione finale)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)
    labels = predictions.label_ids

    # Calcola le metriche
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")


def analyze(text: str) -> dict[str, float]:
    """
    Analizza il testo e restituisce un dizionario con le classifiche.
    """
    # Inizializza la pipeline con il tuo modello addestrato e il tokenizer
    # Assicurati che il percorso in "model" sia quello dove hai salvato il modello
    model_path = "./results"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    # Esegui una previsione
    result = sentiment_pipeline(text)

    # Il risultato sarà una lista di dizionari con label e score
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    if args.train:
        train()
