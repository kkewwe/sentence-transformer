import torch
from fastapi import FastAPI
from pydantic import BaseModel
from app.main import load_model, encode_sentences, get_embedding_dimension, MultiTaskModel, project_embeddings
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

app = FastAPI(
    title="Sentence Transformer Multi-Task API",
    description="API for encoding sentences, classification, and sentiment analysis.",
    version="2.0.0",
)

sentence_model = load_model()
embedding_dim = get_embedding_dimension(sentence_model)

multitask_model = MultiTaskModel(
    embedding_dim=embedding_dim,
    num_classes_task_a=3,
    num_classes_task_b=2
)

class EncodingSentenceRequest(BaseModel):
    sentences: list = ["The quick brown fox jumped over the lazy dog"]
    embedding_dimensions: int = 384

class SentenceRequestA(BaseModel):
    sentences: list = ["Soccer is pretty cool!"]
    labels: List[str] = ["Sports", "Technology", "Food"]

class SentenceRequestB(BaseModel):
    sentences: list = ["The weather is so gloomy"]
    labels: List[str] = ["Positive", "Negative"]

class EmbeddingResponse(BaseModel):
    embeddings: list

class TaskAResponse(BaseModel):
    classification_logits: list
    predicted_labels: list

class TaskBResponse(BaseModel):
    sentiment_logits: list
    predicted_sentiments: list

@app.post("/encode", response_model=EmbeddingResponse)
async def encode_sentences_api(request: EncodingSentenceRequest):
    embeddings = encode_sentences(sentence_model, request.sentences)

    if request.embedding_dimensions != 384:
        embeddings = project_embeddings(embeddings, request.embedding_dimensions)

    embeddings_list = embeddings.detach().cpu().numpy().tolist()
    return EmbeddingResponse(embeddings=embeddings_list)

@app.post("/task-a-classify", response_model=TaskAResponse)
async def classify_sentences_api(request: SentenceRequestA):
    embeddings = encode_sentences(sentence_model, request.sentences)
    task_a_logits, _ = multitask_model(embeddings)
    task_a_logits_list = task_a_logits.detach().cpu().numpy().tolist()

    predictions = torch.argmax(task_a_logits, dim=1)
    predicted_labels = [request.labels[idx] for idx in predictions.tolist()]
    return TaskAResponse(
        classification_logits=task_a_logits_list,
        predicted_labels=predicted_labels
    )

@app.post("/task-b-sentiment", response_model=TaskBResponse)
async def sentiment_sentences_api(request: SentenceRequestB):
    embeddings = encode_sentences(sentence_model, request.sentences)
    _, task_b_logits = multitask_model(embeddings)
    task_b_logits_list = task_b_logits.detach().cpu().numpy().tolist()

    predictions = torch.argmax(task_b_logits, dim=1)
    predicted_sentiments=(request.labels[idx] for idx in predictions.tolist())
    return TaskBResponse(
        sentiment_logits=task_b_logits_list,
        predicted_sentiments=predicted_sentiments
    )
