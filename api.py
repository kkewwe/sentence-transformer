from fastapi import FastAPI
from pydantic import BaseModel
from app.main import load_model, encode_sentences, get_embedding_dimension, MultiTaskModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class SentenceRequest(BaseModel):
    sentences: list

class EmbeddingResponse(BaseModel):
    embeddings: list

class TaskAResponse(BaseModel):
    classification_logits: list

class TaskBResponse(BaseModel):
    sentiment_logits: list

@app.post("/encode", response_model=EmbeddingResponse)
async def encode_sentences_api(request: SentenceRequest):
    embeddings = encode_sentences(sentence_model, request.sentences)
    embeddings_list = embeddings.detach().cpu().numpy().tolist()
    return EmbeddingResponse(embeddings=embeddings_list)

@app.post("/task-a-classify", response_model=TaskAResponse)
async def classify_sentences_api(request: SentenceRequest):
    embeddings = encode_sentences(sentence_model, request.sentences)
    task_a_logits, _ = multitask_model(embeddings)
    task_a_logits_list = task_a_logits.detach().cpu().numpy().tolist()
    return TaskAResponse(classification_logits=task_a_logits_list)

@app.post("/task-b-sentiment", response_model=TaskBResponse)
async def sentiment_sentences_api(request: SentenceRequest):
    embeddings = encode_sentences(sentence_model, request.sentences)
    _, task_b_logits = multitask_model(embeddings)
    task_b_logits_list = task_b_logits.detach().cpu().numpy().tolist()
    return TaskBResponse(sentiment_logits=task_b_logits_list)
