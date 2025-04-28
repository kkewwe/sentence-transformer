import torch
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import logging

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# multi-task model, one head for task a, one for task b
class MultiTaskModel(nn.Module):
    def __init__(self, embedding_dim, num_classes_task_a, num_classes_task_b):
        super(MultiTaskModel, self).__init__()
        self.task_a_classifier = nn.Linear(embedding_dim, num_classes_task_a)
        self.task_b_classifier = nn.Linear(embedding_dim, num_classes_task_b)

    def forward(self, embeddings):
        try:
            output_a = self.task_a_classifier(embeddings)
            output_b = self.task_b_classifier(embeddings)
            return output_a, output_b
        except Exception as e:
            logger.error(f"Error in MultiTaskModel forward pass: {e}")
            raise

# load pre-trained model
def load_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        raise

# encode a list of sentences into embeddings
def encode_sentences(model: SentenceTransformer, sentences: list):
    try:
        logger.info("Encoding sentences...")
        embeddings = model.encode(sentences, convert_to_tensor=True)
        return embeddings
    except Exception as e:
        logger.error(f"Error encoding sentences: {e}")
        raise

# get the embedding dimensions
def get_embedding_dimension(model: SentenceTransformer) -> int:
    try:
        sample_embedding = model.encode(["test"], convert_to_tensor=True)
        return sample_embedding.shape[1]
    except Exception as e:
        logger.error(f"Error getting embedding dimension: {e}")
        raise

# linear projection layer to change dimensions
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        try:
            return self.linear(x)
        except Exception as e:
            logger.error(f"Error in ProjectionLayer forward pass: {e}")
            raise

# project the embeddings (if needed)
def project_embeddings(embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
    try:
        original_dim = embeddings.shape[1]
        if original_dim == target_dim:
            return embeddings
        logger.info(f"Projecting embeddings from {original_dim} -> {target_dim}")
        projection_layer = ProjectionLayer(original_dim, target_dim)
        return projection_layer(embeddings)
    except Exception as e:
        logger.error(f"Error projecting embeddings: {e}")
        raise
