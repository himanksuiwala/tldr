import logging
import ollama
from typing import List

from settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = settings.model_embedding


def get_vector_embeddings(text: str) -> List[float]:
    """
    Generate vector embeddings for the given text using Ollama.
    """
    try:
        logger.debug(f"Generating embeddings for text of length: {len(text)}")

        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        embedding = response["embeddings"][0]

        logger.debug(f"Generated embedding with dimension: {len(embedding)}")
        return embedding

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise Exception(f"Embedding generation failed: {str(e)}")
