from sentence_transformers import SentenceTransformer
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskModel(nn.Module):
    def __init__(self, embedding_dim, num_classes_task_a, num_classes_task_b):
        super(MultiTaskModel, self).__init__()
        self.task_a_classifier = nn.Linear(embedding_dim, num_classes_task_a)
        self.task_b_classifier = nn.Linear(embedding_dim, num_classes_task_b)

    def forward(self, embeddings):
        output_a = self.task_a_classifier(embeddings)
        output_b = self.task_b_classifier(embeddings)
        return output_a, output_b

def load_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model

def encode_sentences(model: SentenceTransformer, sentences: list):
    logger.info("Encoding sentences...")
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings

def get_embedding_dimension(model: SentenceTransformer) -> int:
    sample_embedding = model.encode(["test"], convert_to_tensor=True)
    return sample_embedding.shape[1]

