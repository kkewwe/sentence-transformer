import logging
import random

import numpy as np
import torch
from main import load_model, encode_sentences, get_embedding_dimension, MultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

test_sentences = [
    "Soccer is pretty cool!",
    "Hello Fetch!",
    "The weather is so gloomy."
]
task_a_labels = ["Greetings", "Sports", "Weather"]
task_b_labels = ["Positive", "Negative"]

def test_sentence_embeddings():
    logger.info("Testing sentence embeddings...")
    model = load_model()
    embeddings = encode_sentences(model, test_sentences)

    assert embeddings.shape[0] == len(test_sentences), "Mismatch in number of embeddings"
    logger.info(f"Number of sentences: {len(test_sentences)}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")

    for idx, embed in enumerate(embeddings):
        logger.info(f"Sentence: {test_sentences[idx]}")
        logger.info(f"Embedding shape: {embed.shape}")
        logger.info(f"Embedding snippet: {embed[:5]}...\n")

def test_multitask_pipeline():
    logger.info("Testing multitask pipeline...")
    model = load_model()
    embeddings = encode_sentences(model, test_sentences)
    dimension = get_embedding_dimension(model)

    logger.info(f"Embedding dimension: {dimension}")

    multitask_model = MultiTaskModel(
        embedding_dim=dimension,
        num_classes_task_a=len(task_a_labels),
        num_classes_task_b=len(task_b_labels)
    )

    output_a, output_b = multitask_model(embeddings)

    assert output_a.shape[0] == len(test_sentences), "Mismatch in Task A output batch size"
    assert output_b.shape[0] == len(test_sentences), "Mismatch in Task B output batch size"
    assert output_a.shape[1] == len(task_a_labels), "Mismatch in Task A class count"
    assert output_b.shape[1] == len(task_b_labels), "Mismatch in Task B class count"

    logger.info(f"Task A Output Shape (Classification logits): {output_a.shape}")
    logger.info(f"Task B Output Shape (Sentiment logits): {output_b.shape}")

def test_task_a_classification():
    logger.info("Testing Task A: Sentence Classification...")
    model = load_model()
    embeddings = encode_sentences(model, test_sentences)
    dimension = get_embedding_dimension(model)

    multitask_model = MultiTaskModel(
        embedding_dim=dimension,
        num_classes_task_a=len(task_a_labels),
        num_classes_task_b=len(task_b_labels)
    )

    task_a_logits, _ = multitask_model(embeddings)
    predictions = torch.argmax(task_a_logits, dim=1)
    predicted_labels = [task_a_labels[idx] for idx in predictions.tolist()]

    logger.info(f"Predicted Task A labels: {predicted_labels}")
    assert len(predicted_labels) == len(test_sentences), "Mismatch in number of Task A predictions"

def test_task_b_sentiment_analysis():
    logger.info("Testing Task B: Sentiment Analysis...")
    model = load_model()
    embeddings = encode_sentences(model, test_sentences)
    dimension = get_embedding_dimension(model)

    multitask_model = MultiTaskModel(
        embedding_dim=dimension,
        num_classes_task_a=len(task_a_labels),
        num_classes_task_b=len(task_b_labels)
    )

    _, task_b_logits = multitask_model(embeddings)
    predictions = torch.argmax(task_b_logits, dim=1)
    predicted_sentiments = [task_b_labels[idx] for idx in predictions.tolist()]

    logger.info(f"Predicted sentiments: {predicted_sentiments}")
    assert len(predicted_sentiments) == len(test_sentences), "Mismatch in number of Task B predictions"

if __name__ == "__main__":
    set_seed(1467)
    test_sentence_embeddings()
    test_multitask_pipeline()
    test_task_a_classification()
    test_task_b_sentiment_analysis()
