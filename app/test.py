import logging
from main import load_model, encode_sentences, get_embedding_dimension, MultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello Fetch!",
    "According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small "
    "to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans "
    "think is impossible. Yellow, black. Yellow, black. Yellow, black. Yellow, black. Ooh, black and yellow!",
]

def test_sentence_embeddings():
    model = load_model()
    embeddings = encode_sentences(model, test_sentences)

    for idx, embed in enumerate(embeddings):
        logger.info(f"Sentence: {test_sentences[idx]}")
        logger.info(f"Embedding shape: {embed.shape}")
        logger.info(f"Embedding snippet: {embed[:5]}...\n")

def test_multitask_pipeline():
    model = load_model()
    embeddings = encode_sentences(model, test_sentences)
    dimension = get_embedding_dimension(model)
    logger.info(f"Embedding dimension: {dimension}")

    multitask_model = MultiTaskModel(
        embedding_dim=dimension,
        num_classes_task_a=3,
        num_classes_task_b=2
    )

    output_a, output_b = multitask_model(embeddings)
    logger.info(f"Task A Output Shape (Classification): {output_a.shape}")
    logger.info(f"Task B Output Shape (Sentiment Analysis): {output_b.shape}")

if __name__ == "__main__":
    test_sentence_embeddings()
    test_multitask_pipeline()
